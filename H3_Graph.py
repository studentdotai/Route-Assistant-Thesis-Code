class H3Graph(BaseGraph):
	def __init__(self, enc_schema_name: str, graph_schema_name: str, route_schema_name: str):
		# Initialize the parent BaseGraph class
		# Pass None for points if they're not available at initialization
		super().__init__(
			departure_port=None,
			arrival_port=None,
			port_boundary=None,
			enc_schema_name=enc_schema_name,
			graph_schema_name=graph_schema_name,
		)
		self.route_schema = route_schema_name

	def pg_create_h3_grid(self, base_layer="seaare", optional_layers=None, enc_names: list = None, usage_bands=None,
						  route_buffer=None, save_to_db=False, schema_name=None, table_name="grid_h3"):
		"""
		Creates a grid suitable for H3 cell generation by slicing and combining layer geometries.
		Uses seaare as base layer and filters by usage bands, with optional additional layers.
		Removes land areas from the resulting geometry.

		Parameters:
			base_layer (str): The name of the base table (default is "seaare")
			enc_names (list): List of ENC identifier strings to filter the features.
			optional_layers (list): List of additional layers to include (e.g., ["fairwy", "tsslpt", "prcare"])
			usage_bands (dict): Dictionary mapping layer names to usage bands to include
							   (e.g., {"seaare": ["1", "2"], "fairwy": ["3", "4"]})
			route_buffer (shapely.geometry.Polygon): Optional buffer polygon to restrict the area
			save_to_db (bool): Whether to save the grid to PostGIS (Default is False)
			schema_name (str): Schema where the grid will be saved. If None, uses graph_schema
			table_name (str): Table name for the saved grid (Default is "grid_h3")

		Returns:
			dict: A dictionary mapping keys of the form "<layer>_band<usage_band>" to a GeoJSON string.
		"""
		# Set defaults
		if optional_layers is None:
			optional_layers = ["fairwy", "tsslpt", "prcare"]

		if usage_bands is None:
			usage_bands = {
				"seaare": ["1", "2", "3", "4", "5", "6"],
				"fairwy": ["3", "4", "5"],
				"tsslpt": ["3", "4", "5"],
				"prcare": ["3", "4", "5"]
			}

		# Use the graph_schema as default if schema_name is None
		if schema_name is None:
			schema_name = self.graph_schema

		# If provided, format the ENC names filter as in pg_fine_grid
		formated_names = None
		if enc_names is not None:
			formated_names = self.pg._format_enc_names(enc_names)

		# Initialize results dictionary
		results = {}
		collection = {}

		combined_query = f"""
				WITH combined_seaare AS (
					SELECT ST_Union(
						ST_Intersection(wkb_geometry, ST_GeomFromText('{route_buffer.wkt if route_buffer else "POLYGON EMPTY"}', 4326))
					) as geom
					FROM "{self.enc_schema}"."{base_layer}"
					WHERE substring(dsid_dsnm from 3 for 1) IN ('1','2','3','4','5','6')
					{"AND dsid_dsnm = ANY(:enc_names)" if formated_names else ""}
					{f"AND ST_Intersects(wkb_geometry, ST_GeomFromText('{route_buffer.wkt}', 4326))" if route_buffer else ""}
				),
				land_areas AS (
					SELECT ST_Union(
						ST_Intersection(wkb_geometry, ST_GeomFromText('{route_buffer.wkt if route_buffer else "POLYGON EMPTY"}', 4326))
					) as geom
					FROM "{self.enc_schema}"."lndare"
					WHERE substring(dsid_dsnm from 3 for 1) IN ('4','5','6')
					{"AND dsid_dsnm = ANY(:enc_names)" if formated_names else ""}
					{f"AND ST_Intersects(wkb_geometry, ST_GeomFromText('{route_buffer.wkt}', 4326))" if route_buffer else ""}
				)
				SELECT ST_AsGeoJSON(
					CASE 
						WHEN (SELECT geom FROM land_areas) IS NOT NULL 
						THEN ST_Difference((SELECT geom FROM combined_seaare), (SELECT geom FROM land_areas))
						ELSE (SELECT geom FROM combined_seaare)
					END
				) as geojson
			"""

		with self.pg.connect() as conn:
			params = {}
			if formated_names:
				params["enc_names"] = formated_names
			combined_result = conn.execute(text(combined_query), params).fetchone()

			if combined_result and combined_result[0]:
				results["combined_grid"] = combined_result[0]
			else:
				collection["combined_grid"] = '{"type": "GeometryCollection", "geometries": []}'

		# Process base layer (seaare) by usage bands
		for usage_band in usage_bands.get(base_layer, ["1", "2", "3", "4", "5", "6"]):
			# Build the spatial query with usage band filter and optional ENC filtering
			base_query = f"""
				 WITH base_geometry AS (
					 SELECT ST_Union(
						ST_Intersection(wkb_geometry, ST_GeomFromText('{route_buffer.wkt if route_buffer else "POLYGON EMPTY"}', 4326))
					) as geom
					 FROM "{self.enc_schema}"."{base_layer}"
					 WHERE substring(dsid_dsnm from 3 for 1) = '{usage_band}'
					 {"AND dsid_dsnm = ANY(:enc_names)" if formated_names else ""}
					 {f"AND ST_Intersects(wkb_geometry, ST_GeomFromText('{route_buffer.wkt}', 4326))" if route_buffer else ""}
				 ),
				 land_areas AS (
					 SELECT ST_Union(
						ST_Intersection(wkb_geometry, ST_GeomFromText('{route_buffer.wkt if route_buffer else "POLYGON EMPTY"}', 4326))
					) as geom
					 FROM "{self.enc_schema}"."lndare"
					 WHERE substring(dsid_dsnm from 3 for 1) IN ('4','5','6')
					 {"AND dsid_dsnm = ANY(:enc_names)" if formated_names else ""}
					 {f"AND ST_Intersects(wkb_geometry, ST_GeomFromText('{route_buffer.wkt}', 4326))" if route_buffer else ""}
				 )
				 SELECT ST_AsGeoJSON(
					 CASE 
						 WHEN (SELECT geom FROM land_areas) IS NOT NULL 
						 THEN ST_Difference((SELECT geom FROM base_geometry), (SELECT geom FROM land_areas))
						 ELSE (SELECT geom FROM base_geometry)
					 END
				 ) as geojson
				 """

			with self.pg.connect() as conn:
				params = {}
				if formated_names:
					params["enc_names"] = formated_names
				cur_result = conn.execute(text(base_query), params).fetchone()

				if cur_result and cur_result[0]:
					results[f"{base_layer}_band{usage_band}"] = cur_result[0]
				else:
					collection[f"{base_layer}_band{usage_band}"] = '{"type": "GeometryCollection", "geometries": []}'

		# Process optional layers
		for layer in optional_layers:

			# Build the spatial query for optional layer using the same ENC filtering if provided
			layer_query = f"""
				 WITH layer_geometry AS (
					 SELECT ST_Union(
						ST_Intersection(wkb_geometry, ST_GeomFromText('{route_buffer.wkt if route_buffer else "POLYGON EMPTY"}', 4326))
					 ) as geom
					 FROM "{self.enc_schema}"."{layer}"
					 WHERE substring(dsid_dsnm from 3 for 1) IN ('3','4', '5', '6')
					 {"AND dsid_dsnm = ANY(:enc_names)" if formated_names else ""}
					 {f"AND ST_Intersects(wkb_geometry, ST_GeomFromText('{route_buffer.wkt}', 4326))" if route_buffer else ""}
				 ),
				 land_areas AS (
					 SELECT ST_Union(
						ST_Intersection(wkb_geometry, ST_GeomFromText('{route_buffer.wkt if route_buffer else "POLYGON EMPTY"}', 4326))
					 ) as geom
					 FROM "{self.enc_schema}"."lndare"
					 WHERE substring(dsid_dsnm from 3 for 1) IN ('4','5','6')
					 {"AND dsid_dsnm = ANY(:enc_names)" if formated_names else ""}
					 {f"AND ST_Intersects(wkb_geometry, ST_GeomFromText('{route_buffer.wkt}', 4326))" if route_buffer else ""}
				 )
				 SELECT ST_AsGeoJSON(
					 CASE 
						 WHEN (SELECT geom FROM land_areas) IS NOT NULL 
						 THEN ST_Difference((SELECT geom FROM layer_geometry), (SELECT geom FROM land_areas))
						 ELSE (SELECT geom FROM layer_geometry)
					 END
				 ) as geojson
				 """

			with self.pg.connect() as conn:
				params = {}
				if formated_names:
					params["enc_names"] = formated_names
				cur_result = conn.execute(text(layer_query), params).fetchone()

				if cur_result and cur_result[0] and cur_result[0] != '{"type":"GeometryCollection","geometries":[]}':
					results[f"{layer}"] = cur_result[0]
				else:
					collection[f"{layer}"] = '{"type": "GeometryCollection", "geometries": []}'

		# Save to database if requested
		if save_to_db and results:
			create_schema_sql = f'CREATE SCHEMA IF NOT EXISTS "{schema_name}";'
			drop_table_sql = f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}"'

			create_table_sql = f"""
				 CREATE TABLE IF NOT EXISTS "{schema_name}"."{table_name}" (
				 id SERIAL PRIMARY KEY,
				 layer_name VARCHAR(50),
				 usage_band VARCHAR(10),
				 grid GEOMETRY(Geometry,4326),
				 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
				 );
				 """

			base_table = f'{table_name}_base'
			# SQL for combined grid table
			drop_combined_table_sql = f'DROP TABLE IF EXISTS "{schema_name}"."{base_table}"'
			create_combined_table_sql = f"""
					 CREATE TABLE IF NOT EXISTS "{schema_name}"."{base_table}" (
						 id SERIAL PRIMARY KEY,
						 grid GEOMETRY(Geometry,4326),
						 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
					 );
					 """

			with self.pg.connect() as conn:
				conn.execute(text(create_schema_sql))

				# Create regular grid table
				conn.execute(text(drop_table_sql))
				conn.execute(text(create_table_sql))

				# Create combined grid table
				conn.execute(text(drop_combined_table_sql))
				conn.execute(text(create_combined_table_sql))

				# Insert each result into the regular table (except combined grid)
				for key, geojson in results.items():
					if key != "combined_grid":
						# Parse the key to get layer name and usage band
						if "_band" in key:
							parts = key.split('_band')
							layer_name = parts[0]
							usage_band = parts[1]
						else:
							layer_name = key
							usage_band = ''

						insert_sql = f"""
								 INSERT INTO "{schema_name}"."{table_name}" (layer_name, usage_band, grid)
								 VALUES (:layer_name, :usage_band, ST_GeomFromGeoJSON(:geojson));
								 """

						conn.execute(text(insert_sql), {
							"layer_name": layer_name,
							"usage_band": usage_band,
							"geojson": geojson
						})

				# Insert combined grid into the dedicated table
				if "combined_grid" in results:
					insert_combined_sql = f"""
							 INSERT INTO "{schema_name}"."{base_table}" (grid)
							 VALUES (ST_GeomFromGeoJSON(:geojson));
							 """
					conn.execute(text(insert_combined_sql), {
						"geojson": results["combined_grid"]
					})

				conn.commit()

			print(f"H3 grid layers saved to PostGIS in {schema_name}.{table_name}")
			print(f"Combined H3 grid saved to PostGIS in {schema_name}.{base_table}")

		return results

	def create_h3_graph(self, base_resolution=7, detail_resolution=11,
					   base_layer="seaare", optional_layers=None, enc_names: list = None, usage_bands=None,
					   route_buffer=None, save_to_db=False, table_name="grid_h3"):
		"""
		Creates a multi-resolution H3 grid based on maritime features.

		Parameters:
			base_resolution (int): H3 resolution for base areas (5-6 recommended)
			detail_resolution (int): H3 resolution for detailed areas (7-9 recommended)
			base_layer (str): The name of the base table (default is "seaare")
			optional_layers (list): List of additional layers for detailed resolution
			enc_names (list): List of ENC identifier strings to filter the features.
			usage_bands (dict): Dictionary mapping layer names to usage bands
			route_buffer (shapely.geometry.Polygon): Optional buffer to restrict the area
			save_to_db (bool): Whether to save the H3 cells to PostGIS
			table_name (str): Table name for the saved grid (Default is "grid_h3")
		Returns:
			nx.Graph: NetworkX graph built from the multi-resolution H3 grid
		"""

		print(f"{datetime.now()} - Starting H3 grid creation")

		# Retrieve grid polygons for different layers
		grid_geojsons = self.pg_create_h3_grid(
			base_layer=base_layer,
			optional_layers=optional_layers,
			enc_names = enc_names,
			usage_bands=usage_bands,
			route_buffer=route_buffer,
			save_to_db=save_to_db,
			table_name = table_name
		)
		print(f"{datetime.now()} - Retrieved {len(grid_geojsons)} grid polygons")

		# Initialize sets for H3 cells
		base_hexagons = set()
		detail_hexagons = set()

		# Process each polygon only once, differentiating base and detail layers.
		for key, geojson_str in grid_geojsons.items():
			if key != "combined_grid":
				try:
					geojson = json.loads(geojson_str)

					if key.startswith(base_layer):
						band = key.split('_')[1]
						print(f"Band: {band}")
						if band in ["band1", "band2"]:
							cells = h3.polygon_to_cells(h3.geo_to_h3shape(geojson), base_resolution)
						elif band == "band3":
							cells = h3.polygon_to_cells(h3.geo_to_h3shape(geojson), detail_resolution - 2)
						elif band == "band4":
							cells = h3.polygon_to_cells(h3.geo_to_h3shape(geojson), detail_resolution - 1)
						else:
							cells = h3.polygon_to_cells(h3.geo_to_h3shape(geojson), detail_resolution)
						base_hexagons.update(cells)
						print(f"{datetime.now()} - Added {len(cells)} base cells from {key}")
					elif key in ["prcare"]:
						cells = h3.polygon_to_cells(h3.geo_to_h3shape(geojson), detail_resolution - 1)
						detail_hexagons.update(cells)
						print(f"{datetime.now()} - Added {len(cells)} detail cells from {key}")
					else:
						cells = h3.polygon_to_cells(h3.geo_to_h3shape(geojson), detail_resolution)
						detail_hexagons.update(cells)
						print(f"{datetime.now()} - Added {len(cells)} detail cells from {key}")
				except Exception as e:
					print(f"Error processing {key}: {str(e)}")

		# Remove base cells that are covered by detail cells using a set comprehension
		print(f'Cleaning Child - Parrent Cells')
		base_to_remove = set()
		for detail_cell in detail_hexagons:
			# Get the cell's current resolution
			current_res = h3.get_resolution(detail_cell)

			# Find the appropriate parent at base_resolution
			if current_res > base_resolution:
				parent = h3.cell_to_parent(detail_cell, base_resolution)
				base_to_remove.add(parent)

		# Remove the identified base cells
		base_hexagons -= base_to_remove

		print(
			f"{datetime.now()} - Final grid has {len(base_hexagons)} base cells and {len(detail_hexagons)} detail cells")

		# Create graph from hexagons
		G = nx.Graph()

		# Helper: cache cell centers for reuse
		def get_center(cell):
			# Returns (lng, lat) as used by the graph nodes
			lat, lng = h3.cell_to_latlng(cell)
			return (lng, lat)

		# Add nodes for base and detail resolution
		for h3_idx in base_hexagons:
			G.add_node(get_center(h3_idx), h3_index=h3_idx, resolution=base_resolution)
		for h3_idx in detail_hexagons:
			G.add_node(get_center(h3_idx), h3_index=h3_idx, resolution=detail_resolution)

		print(f"{datetime.now()} - Added {len(G.nodes)} nodes to graph")

		edges_added = 0

		# Function to add an edge between two cells given their H3 indexes and centers.
		def add_edge(cell_a, cell_b):
			center_a = get_center(cell_a)
			center_b = get_center(cell_b)
			weight = Misceleaneous.haversine(center_a[0], center_a[1], center_b[0], center_b[1])
			G.add_edge(center_a, center_b, weight=weight, h3_edge=(cell_a, cell_b))

		# Add edges for base resolution cells
		for h3_idx in base_hexagons:
			for neighbor in h3.grid_ring(h3_idx, 1):
				if neighbor in base_hexagons:
					add_edge(h3_idx, neighbor)
					edges_added += 1

		# Add edges for detail resolution cells
		for h3_idx in detail_hexagons:
			for neighbor in h3.grid_ring(h3_idx, 1):
				if neighbor in detail_hexagons:
					add_edge(h3_idx, neighbor)
					edges_added += 1

		# Optimized: Connect cross-resolution edges by iterating over detail cells only.
		# For each detail cell, compute its parent at the base resolution, then check the neighbors
		# of that parent. If a neighboring base cell exists, add an edge.
		for detail_idx in detail_hexagons:
			detail_parent = h3.cell_to_parent(detail_idx, base_resolution)
			# Get neighbors (fixed, at most 6) of the parent cell
			parent_neighbors = h3.grid_ring(detail_parent, 1)
			for base_candidate in parent_neighbors:
				if base_candidate in base_hexagons:
					add_edge(base_candidate, detail_idx)
					edges_added += 1

		print(f"{datetime.now()} - Added a total of {edges_added} edges to graph")

		return G, grid_geojsons['combined_grid']

	
