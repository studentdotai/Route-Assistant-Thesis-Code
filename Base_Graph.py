class BaseGraph:
	"""
	The class RouteAssistant wraps the entire workflow starting from base grid creation.
	• The create_base_grid method executes the SQL (with extra combined grid) and saves grids to PostGIS if required.
	• The create_grid_graph method builds the grid graph (by merging main and extra grids) using the helper create_grid_subgraph_v2.
	• Methods save_graph, clean_graph, and load_graph handle PostGIS persistence and reloading of the graph.
	• The route method computes the first route using A* (through the astar_r
	"""


	def __init__(self, departure_port, arrival_port, port_boundary, enc_schema_name: str, graph_schema_name:str):
		# Use the global PostGIS connection and Miscellaneous conversion functions.
		self.pg = PostGIS()
		self.misc = Miscellaneous()
		self.departure_point = departure_port
		self.arrival_point = arrival_port
		self.port_boundary = port_boundary
		self.enc_schema = enc_schema_name
		self.graph_schema = graph_schema_name



	def create_base_grid(self, layer_table="seaare", extra_grids=["fairwy", "tsslpt", "prcare"],
						 reduce_distance=2, save_to_db=True):
		"""
		Creates a base grid over the port boundary (using main and extra grids) and returns GeoJSON
		for points (departure, start, end, arrival) as well as the main grid and extra grid.
		Additionally, the combined grid is saved to PostGIS (if save_to_db is True) for later use.
		"""
		reduce_distance = self.misc.miles_to_decimal(reduce_distance)

		# Build dynamic UNION ALL query for extra grids
		extra_grid_union = " UNION ALL ".join(
			[f"SELECT wkb_geometry AS geom, dsid_dsnm, '{table}' AS grid_name FROM \"{self.enc_schema}\".\"{table}\""
			 for table in extra_grids]
		)

		with self.pg.connect() as conn:
			grid_query = text(f"""
			WITH grid_enc AS (
				SELECT ST_Union(s.wkb_geometry) AS grid_geom
				FROM "{self.enc_schema}"."{layer_table}" s
				WHERE substring(s.dsid_dsnm from 3 for 1)  IN ('1','2')
				  AND ST_Intersects(s.wkb_geometry, ST_GeomFromText(:port_boundary_geom, 4326))
			), reduced_grid AS (
				SELECT CASE 
						 WHEN :reduce_distance > 0 THEN ST_Buffer(grid_geom, -:reduce_distance)
						 ELSE grid_geom
					   END AS grid_geom
				FROM grid_enc
			), extra_grid AS (
				SELECT grid_name, ST_Union(tbl.geom) AS grid_geom
				FROM (
					{extra_grid_union}
				) tbl
				WHERE substring(tbl.dsid_dsnm from 3 for 1) IN ('4','3')
				  AND ST_Intersects(tbl.geom, ST_GeomFromText(:port_boundary_geom, 4326))
				GROUP BY grid_name
			), combined_grid AS (
				SELECT ST_Union(rg.grid_geom, eg.grid_geom) AS grid_geom
				FROM reduced_grid rg, extra_grid eg
			), dumped AS (
				SELECT (dp).geom AS comp
				FROM (SELECT ST_Dump(combined_grid.grid_geom) AS dp FROM combined_grid) d
			), connected AS (
				SELECT d1.comp
				FROM dumped d1
				WHERE EXISTS (
					 SELECT 1 FROM dumped d2 
					 WHERE ST_DWithin(d1.comp, d2.comp, 0.0001) AND ST_Area(d1.comp) > 0.01
				)
			), filtered_grid AS (
				SELECT ST_Union(comp) AS grid_geom
				FROM connected
			), adjusted_points AS (
				SELECT
					CASE 
						WHEN ST_Contains(fg.grid_geom, ST_GeomFromText(:departure_point, 4326))
						  THEN ST_GeomFromText(:departure_point, 4326)
						ELSE ST_ClosestPoint(fg.grid_geom, ST_GeomFromText(:departure_point, 4326))
					END AS start_point,
					CASE 
						WHEN ST_Contains(fg.grid_geom, ST_GeomFromText(:arrival_point, 4326))
						  THEN ST_GeomFromText(:arrival_point, 4326)
						ELSE ST_ClosestPoint(fg.grid_geom, ST_GeomFromText(:arrival_point, 4326))
					END AS end_point,
					fg.grid_geom
				FROM filtered_grid fg
			)
			SELECT 
				ST_AsGeoJSON(ST_GeomFromText(:departure_point, 4326)) AS departure_point,
				ST_AsGeoJSON(ap.start_point) AS start_point,
				ST_AsGeoJSON(ap.end_point) AS end_point,
				ST_AsGeoJSON(ST_GeomFromText(:arrival_point, 4326)) AS arrival_point,
				ST_AsGeoJSON(rg.grid_geom) AS main_grid_geojson,
				ST_AsGeoJSON(ST_Union(eg.grid_geom)) AS extra_grid,
				ST_AsGeoJSON(fg.grid_geom) AS combined_grid
			FROM adjusted_points ap, reduced_grid rg, extra_grid eg, filtered_grid fg
			GROUP BY departure_point, ap.start_point, ap.end_point, arrival_point, rg.grid_geom, fg.grid_geom;
			""")

			params = {
				'port_boundary_geom': self.port_boundary.wkt if hasattr(self.port_boundary,
																		'wkt') else self.port_boundary,
				'reduce_distance': reduce_distance,
				'departure_point': self.departure_point.wkt if hasattr(self.departure_point, 'wkt') else self.departure_point,
				'arrival_point': self.arrival_point.wkt if hasattr(self.arrival_point, 'wkt') else self.arrival_point,
			}

			result = conn.execute(grid_query, params)
			row = result.fetchone()

			final_result = {
				"points": {
					"dep_point": row[0],
					"start_point": row[1],
					"end_point": row[2],
					"arr_point": row[3]
				},
				"main_grid": row[4],
				"extra_grids": row[5],
				"combined_grid": row[6]
			}

			if save_to_db:
				schema = self.graph_schema
				table_main = "grid_main"
				table_extra = "grid_extra"
				table_combined = "grid_combined"
				create_schema_sql = f'CREATE SCHEMA IF NOT EXISTS "{schema}";'

				drop_table_main_sql = 'DROP TABLE IF EXISTS "{}"."grid_main"'.format(schema)
				drop_table_extra_sql = 'DROP TABLE IF EXISTS "{}"."grid_extra"'.format(schema)
				drop_table_combined_sql = 'DROP TABLE IF EXISTS "{}"."grid_combined"'.format(schema)

				create_table_main_sql = f"""
					CREATE TABLE IF NOT EXISTS "{schema}"."{table_main}" (
						 id SERIAL PRIMARY KEY,
						 grid GEOMETRY(MultiPolygon,4326)
					);
					"""
				create_table_extra_sql = f"""
					CREATE TABLE IF NOT EXISTS "{schema}"."{table_extra}" (
						id SERIAL PRIMARY KEY,
						grid GEOMETRY(MultiPolygon,4326)
					);
					"""
				create_table_combined_sql = f"""
					CREATE TABLE IF NOT EXISTS "{schema}"."{table_combined}" (
						id SERIAL PRIMARY KEY,
						grid GEOMETRY(MultiPolygon,4326)
					);
					"""
				insert_main_sql = f"""
					INSERT INTO "{schema}"."{table_main}" (grid)
					VALUES (ST_GeomFromGeoJSON(:geojson));
					"""
				insert_extra_sql = f"""
					INSERT INTO "{schema}"."{table_extra}" (grid)
					VALUES (ST_GeomFromGeoJSON(:geojson));
					"""
				insert_combined_sql = f"""
					INSERT INTO "{schema}"."{table_combined}" (grid)
					VALUES (ST_GeomFromGeoJSON(:geojson));
					"""

				with self.pg.connect() as conn:
					conn.execute(text(create_schema_sql))

					conn.execute(text(drop_table_main_sql))
					conn.execute(text(drop_table_extra_sql))
					conn.execute(text(drop_table_combined_sql))

					conn.execute(text(create_table_main_sql))
					conn.execute(text(create_table_extra_sql))
					conn.execute(text(create_table_combined_sql))
					conn.execute(text(insert_main_sql), {"geojson": row[4]})
					conn.execute(text(insert_extra_sql), {"geojson": row[5]})
					conn.execute(text(insert_combined_sql), {"geojson": row[6]})
					conn.commit()
				print(f"Saved main_grid, extra_grid and combined_grid to PostGIS in schema '{schema}'.")
			return final_result

	def create_base_graph(self, grid_geojson, spacings: float = None) -> nx.Graph:
		"""
		Constructs multiple graphs from grid GeoJSONs from the base grid creation.
		Merges the main grid and extra grids using a mesh grid approach defined in create_grid_subgraph_v2.
		"""
		if spacings:
			spacing = spacings
		else:
			spacing = 0.1

		# Handle both string and dictionary inputs
		if isinstance(grid_geojson, str):
			grid = json.loads(grid_geojson)
		elif isinstance(grid_geojson, dict):
			# If it's a dict from create_base_grid, it might contain GeoJSON strings
			if any(key in grid_geojson for key in ['main_grid', 'combined_grid']):
				# Use combined_grid if available, otherwise main_grid
				grid_key = 'combined_grid' if 'combined_grid' in grid_geojson else 'main_grid'
				grid = json.loads(grid_geojson[grid_key])
			else:
				# Assume it's already a parsed GeoJSON object
				grid = grid_geojson
		else:
			raise TypeError("grid_geojson must be either a GeoJSON string or a dictionary")



		if grid["type"] == "Polygon":
			polygon = Polygon(grid['coordinates'][0])
		elif grid["type"] == "MultiPolygon":
			polygon = MultiPolygon([Polygon(coords[0]) for coords in grid['coordinates']])
		else:
			raise ValueError("Invalid GeoJSON type. Expected 'Polygon' or 'MultiPolygon'.")
		print(f"{datetime.now()} - Subgraph v1 Started")

		graph = self.create_grid_subgraph(polygon, spacing)
		print(f"{datetime.now()} - Subgraph v1 Completed")
		return graph

	def create_grid_subgraph(self, polygon, spacing, max_edge_factor=3):
		"""
		Creates a graph for a single grid with specified spacing, using a maximum edge length
		threshold (e.g. 1.5x the spacing) to limit connectivity, thereby avoiding expensive
		spatial operations.

		Parameters:
			polygon (shapely.geometry.Polygon): Polygon geometry for the grid.
			spacing (float): Grid spacing in degrees.
			max_edge_factor (float): Maximum allowed edge length relative to spacing (e.g. 1.5 or 2).

		Returns:
			networkx.Graph: Graph for the specified grid.
		"""
		# Get polygon bounds
		minx, miny, maxx, maxy = polygon.bounds

		# Generate grid points using numpy vectorized arrays
		x_coords, y_coords = np.meshgrid(
			np.arange(minx, maxx + spacing, spacing),
			np.arange(miny, maxy + spacing, spacing)
		)
		print(f"{datetime.now()} - NP Mesh Created \nSpacing: {spacing}")
		# Flatten the meshgrid into coordinate pairs
		points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
		print(f"{datetime.now()} - Column Stack Created")


		# Build nodes only if they fall inside the polygon (OLD APPROACH)
		# nodes = {tuple(pt): Point(pt) for pt in points if polygon.contains(Point(pt))}

		# NEW Use shapely.vectorized.contains to get a boolean mask for points inside the polygon
		mask = contains(polygon, points[:, 0], points[:, 1])
		valid_points = points[mask]

		# Build nodes dictionary (each valid point becomes a node)
		nodes = {tuple(pt): Point(pt) for pt in valid_points}

		print(f"{datetime.now()} - Nodes created: {len(nodes)}")
		# Create a graph and add the nodes
		G = nx.Graph()
		G.add_nodes_from(nodes.keys())
		print(f"{datetime.now()} - Nodes added to Graph")
		# Define eight neighbor directions
		directions = np.array([
			(-spacing, 0), (spacing, 0),
			(0, -spacing), (0, spacing),
			(-spacing, -spacing), (-spacing, spacing),
			(spacing, -spacing), (spacing, spacing)
		])

		# Compute maximum allowed edge length
		max_edge_length = spacing * max_edge_factor

		print(f"{datetime.now()} - Edge Creation Started")
		# Iterate through nodes and add edges if the neighbor exists and distance is within threshold
		for (x, y) in nodes.keys():
			# Build potential neighbor coordinates
			neighbors = [(x + dx, y + dy) for dx, dy in directions if (x + dx, y + dy) in nodes]
			if not neighbors:
				continue  # Skip if no valid neighbors
			# Compute distances vectorized for all neighbors
			distances = np.sqrt(np.sum((np.array(neighbors) - np.array([x, y])) ** 2, axis=1))
			# Filter edges that are within the threshold distance
			valid_edges = [((x, y), nb, {"weight": d}) for nb, d in zip(neighbors, distances) if d <= max_edge_length]
			G.add_edges_from(valid_edges)
		print(f"{datetime.now()} - Edge Creation Complete")
		return G

	def create_grid_subgraph_v2(self, polygon, spacing, max_edge_factor=5, precision=0.01):
		"""
		Creates a graph for a single grid with specified spacing, using a maximum edge length
		threshold (e.g. 1.5x the spacing) to limit connectivity, thereby avoiding expensive
		spatial operations.

		Parameters:
			polygon (shapely.geometry.Polygon): Polygon geometry for the grid.
			spacing (float): Grid spacing in degrees.
			max_edge_factor (float): Maximum allowed edge length relative to spacing (e.g. 1.5 or 2).
			precision (float): Precision for coordinate snapping to avoid floating point issues.

		Returns:
			networkx.Graph: Graph for the specified grid.
		"""
		# Get polygon bounds
		minx, miny, maxx, maxy = polygon.bounds

		# Generate grid points using numpy vectorized arrays
		x_coords, y_coords = np.meshgrid(
			np.arange(minx, maxx + spacing, spacing),
			np.arange(miny, maxy + spacing, spacing)
		)
		# Flatten the meshgrid into coordinate pairs
		points = np.column_stack([x_coords.ravel(), y_coords.ravel()])

		# Snap each coordinate to the fixed precision
		points = np.round(points / precision) * precision

		# Build nodes only if they fall inside the polygon
		nodes = {tuple(pt): Point(pt) for pt in points if polygon.contains(Point(pt))}
		print(f"{datetime.now()} - Nodes found: {len(nodes)}")

		# Create a graph and add the nodes
		G = nx.Graph()
		G.add_nodes_from(nodes.keys())

		# Define eight neighbor directions
		directions = np.array([
			(-spacing, 0), (spacing, 0),
			(0, -spacing), (0, spacing),
			(-spacing, -spacing), (-spacing, spacing),
			(spacing, -spacing), (spacing, spacing)
		])

		# Compute maximum allowed edge length
		max_edge_length = spacing * max_edge_factor

		print(f"{datetime.now()} - Edge Creation Started")
		# Iterate through nodes and add edges if the neighbor exists and distance is within threshold
		for (x, y) in nodes.keys():
			# Build potential neighbor coordinates
			neighbors = [(x + dx, y + dy) for dx, dy in directions if (x + dx, y + dy) in nodes]
			if not neighbors:
				continue  # Skip if no valid neighbors
			# Compute distances vectorized for all neighbors
			distances = np.sqrt(np.sum((np.array(neighbors) - np.array([x, y])) ** 2, axis=1))
			# Filter edges that are within the threshold distance
			valid_edges = [((x, y), nb, {"weight": d}) for nb, d in zip(neighbors, distances) if d <= max_edge_length]
			G.add_edges_from(valid_edges)
		print(f"{datetime.now()} - Edge Creation Complete")
		return G

	def save_graph(self, graph: nx.Graph,
				   nodes_table: str = "graph_nodes", edges_table: str = "graph_edges"):
		"""
		Loads the provided graph into PostGIS by creating nodes and edges tables.
		"""
		print(f"Graph Saved to {nodes_table} and {edges_table} tables")

		create_schema_sql = f'CREATE SCHEMA IF NOT EXISTS "{self.graph_schema}";'
		drop_table_nodes_sql = f"""DROP TABLE IF EXISTS "{self.graph_schema}"."{nodes_table}" """
		drop_table_edges_sql = f"""DROP TABLE IF EXISTS "{self.graph_schema}"."{edges_table}" """

		create_nodes_table_sql = f"""
			CREATE TABLE IF NOT EXISTS "{self.graph_schema}"."{nodes_table}" (
				 id SERIAL PRIMARY KEY,
				 node VARCHAR,
				 geom GEOMETRY(Point,4326)
			);
			"""
		create_edges_table_sql = f"""
			CREATE TABLE IF NOT EXISTS "{self.graph_schema}"."{edges_table}" (
				 id SERIAL PRIMARY KEY,
				 source VARCHAR, -- String representation, Used for NetworkX Graph
				 target VARCHAR,
				 source_id INTEGER REFERENCES "{self.graph_schema}"."{nodes_table}"(id),  -- FK reference, to provide faster node reference  
				 target_id INTEGER REFERENCES "{self.graph_schema}"."{nodes_table}"(id),  
				 weight DOUBLE PRECISION,
				 geom GEOMETRY(LineString,4326)
			);
			"""
		# Prepare node data
		nodes_data = []
		for node in graph.nodes():
			nodes_data.append((str(node), Point(node)))

		# Create lookup dictionary for node IDs
		node_to_id = {}

		with self.pg.connect() as conn:
			conn.execute(text(create_schema_sql))
			conn.execute(text(drop_table_edges_sql))
			conn.execute(text(drop_table_nodes_sql))
			conn.execute(text(create_nodes_table_sql))
			conn.execute(text(create_edges_table_sql))

			# Insert nodes and collect their generated IDs
			for node_val, point in nodes_data:
				insert_node_sql = f"""
						INSERT INTO "{self.graph_schema}"."{nodes_table}" (node, geom)
						VALUES (:node, ST_GeomFromText(:wkt, 4326))
						RETURNING id;
						"""
				node_id = conn.execute(text(insert_node_sql),
									   {"node": node_val, "wkt": point.wkt}).fetchone()[0]
				node_to_id[node_val] = node_id

			# Insert edges with both string and ID references
			for u, v, data in graph.edges(data=True):
				line = LineString([u, v])
				weight = data.get('weight', 0)
				source_str = str(u)
				target_str = str(v)
				source_id = node_to_id[source_str]
				target_id = node_to_id[target_str]

				insert_edge_sql = f"""
						INSERT INTO "{self.graph_schema}"."{edges_table}" 
						(source, target, source_id, target_id, weight, geom)
						VALUES (:source, :target, :source_id, :target_id, :weight, 
								ST_GeomFromText(:wkt, 4326));
						"""
				conn.execute(text(insert_edge_sql),
							 {"source": source_str, "target": target_str,
							  "source_id": source_id, "target_id": target_id,
							  "weight": weight, "wkt": line.wkt})
			conn.commit()
		print(f"Graph saved into schema '{self.graph_schema}': {len(nodes_data)} nodes, {len(graph.edges())} edges.")

	def clean_graph(self, nodes_table: str = "graph_nodes", edges_table: str = "graph_edges",
					grid_table: str = "grid_combined",
					distance_threshold: float = 5, rearrang_graph: bool = False):
		"""
		Cleans graph edges in PostGIS by removing nodes outside the grid and rearranging node connections for close nodes.
		rearange_graph: bool = False, expensive operation, use with caution
		"""
		distance = self.misc.miles_to_decimal(distance_threshold)
		delete_edges_sql = f"""
				DELETE FROM "{self.graph_schema}"."{edges_table}"
				WHERE source_id IN (
					SELECT id FROM "{self.graph_schema}"."{nodes_table}"
					WHERE NOT EXISTS (
						SELECT 1
						FROM "{self.graph_schema}"."{grid_table}" mg
						WHERE ST_Contains(mg.grid, "{nodes_table}".geom)
					)
				) OR target_id IN (
					SELECT id FROM "{self.graph_schema}"."{nodes_table}"
					WHERE NOT EXISTS (
						SELECT 1
						FROM "{self.graph_schema}"."{grid_table}" mg
						WHERE ST_Contains(mg.grid, "{nodes_table}".geom)
					)
				);
			"""

		# Then delete nodes outside the grid
		delete_nodes_sql = f"""
				DELETE FROM "{self.graph_schema}"."{nodes_table}"
				WHERE NOT EXISTS (
					SELECT 1
					FROM "{self.graph_schema}"."{grid_table}" mg
					WHERE ST_Contains(mg.grid, "{nodes_table}".geom)
				);
			"""

		with self.pg.connect() as conn:
			# First delete the edges to maintain referential integrity
			conn.execute(text(delete_edges_sql))
			# Then delete the nodes
			conn.execute(text(delete_nodes_sql))
			conn.commit()
		print("Edges not within the grid have been removed.")
		print("Nodes outside the grid have been removed.")


		if rearrang_graph:
			insert_sql = f"""
				INSERT INTO "{self.graph_schema}"."{edges_table}" (source, target, weight, geom)
				SELECT n1.node::text AS source, n2.node::text AS target,
					   ST_Distance(n1.geom, n2.geom) AS weight,
					   ST_MakeLine(n1.geom, n2.geom) AS geom
				FROM "{self.graph_schema}"."{nodes_table}" n1, "{self.graph_schema}"."{nodes_table}" n2
				WHERE n1.node != n2.node
				  AND ST_Distance(n1.geom, n2.geom) < :dist_threshold
				  AND NOT EXISTS (
						SELECT 1
						FROM "{self.graph_schema}"."{edges_table}" e
						WHERE (e.source = n1.node::text AND e.target = n2.node::text)
						   OR (e.source = n2.node::text AND e.target = n1.node::text)
				  );
				"""
			with self.pg.connect() as conn:
				conn.execute(text(insert_sql), {"dist_threshold": distance})
				conn.commit()
			print("Node connections have been rearranged based on proximity.")

		# Remove orphan nodes(nodes without connected edges)
		delete_orphans_sql = f"""
					DELETE FROM "{self.graph_schema}"."{nodes_table}" n
					WHERE NOT EXISTS (
						SELECT 1 FROM "{self.graph_schema}"."{edges_table}" e
						WHERE e.source_id = n.id
					)
					AND NOT EXISTS (
						SELECT 1 FROM "{self.graph_schema}"."{edges_table}" e
						WHERE e.target_id = n.id
					);
					"""
		with self.pg.connect() as conn:
			conn.execute(text(delete_orphans_sql))
			conn.commit()
		print("Orphan nodes (nodes without connected edges) have been removed.")

	def clean_graph_h3(self, nodes_table: str = "graph_nodes", edges_table: str = "graph_edges",
					grid_table: str = "grid_combined", land_table: str = "lndare",
					usage_bands: list = ["4", "5", "6"]):
		"""
		Efficiently cleans graph by removing:
		1. Nodes outside the grid
		2. Edges intersecting with land polygons
		3. Orphan nodes (nodes without connected edges)

		Parameters:
			nodes_table (str): Name of nodes table in graph schema
			edges_table (str): Name of edges table in graph schema
			grid_table (str): Name of grid table in graph schema
			land_table (str): Name of land area table in ENC schema
			usage_bands (list): List of usage bands to consider for land areas. Default is ["4", "5", "6"]
		"""
		usage_bands_str = "', '".join(usage_bands)

		# 1. Delete edges with nodes outside the grid
		delete_edges_outside_grid_sql = f"""
			DELETE FROM "{self.graph_schema}"."{edges_table}"
			WHERE source_id IN (
				SELECT id FROM "{self.graph_schema}"."{nodes_table}"
				WHERE NOT EXISTS (
					SELECT 1
					FROM "{self.graph_schema}"."{grid_table}" mg
					WHERE ST_Contains(mg.grid, "{nodes_table}".geom)
				)
			) OR target_id IN (
				SELECT id FROM "{self.graph_schema}"."{nodes_table}"
				WHERE NOT EXISTS (
					SELECT 1
					FROM "{self.graph_schema}"."{grid_table}" mg
					WHERE ST_Contains(mg.grid, "{nodes_table}".geom)
				)
			);
		"""

		# 2. Delete nodes outside the grid
		delete_nodes_outside_grid_sql = f"""
			DELETE FROM "{self.graph_schema}"."{nodes_table}"
			WHERE NOT EXISTS (
				SELECT 1
				FROM "{self.graph_schema}"."{grid_table}" mg
				WHERE ST_Contains(mg.grid, "{nodes_table}".geom)
			);
		"""

		# 3. Delete edges intersecting with land areas
		delete_land_intersection_sql = f"""
			DELETE FROM "{self.graph_schema}"."{edges_table}" e
			WHERE EXISTS (
				SELECT 1 
				FROM "{self.enc_schema}"."{land_table}" l
				WHERE substring(l.dsid_dsnm from 3 for 1) IN ('{usage_bands_str}')
				AND ST_Intersects(e.geom, l.wkb_geometry)
			);
		"""

		# 4. Remove orphan nodes (nodes without connected edges)
		delete_orphans_sql = f"""
			DELETE FROM "{self.graph_schema}"."{nodes_table}" n
			WHERE NOT EXISTS (
				SELECT 1 FROM "{self.graph_schema}"."{edges_table}" e
				WHERE e.source_id = n.id
			)
			AND NOT EXISTS (
				SELECT 1 FROM "{self.graph_schema}"."{edges_table}" e
				WHERE e.target_id = n.id
			);
		"""

		with self.pg.connect() as conn:
			# First delete edges that would cause orphan nodes
			conn.execute(text(delete_edges_outside_grid_sql))
			print("Edges with nodes outside the grid have been removed.")

			# Then delete nodes outside the grid
			conn.execute(text(delete_nodes_outside_grid_sql))
			print("Nodes outside the grid have been removed.")

			# Delete edges intersecting with land
			conn.execute(text(delete_land_intersection_sql))
			print("Edges intersecting with land areas have been removed.")

			# Finally, remove any orphaned nodes
			conn.execute(text(delete_orphans_sql))
			print("Orphan nodes (nodes without connected edges) have been removed.")

			conn.commit()

	def pg_get_graph_nodes_edges(self, schema_name=None, graph_name=None, weighted=False, all_columns: bool = False):
		"""
		Connects to PostGIS and retrieves nodes and edges as GeoJSON.

		Parameters:
			schema_name (str, optional): Schema to use. Defaults to self.graph_schema.
			graph_name (str, optional): Suffix for the graph tables. Defaults to using the base graph.
			weighted (bool, optional): If True, fetches the adjusted_weight for edges.
			all_columns (bool, optional): If True, retrieves all columns in the nodes and edges tables.

		Returns:
		  tuple: (nodes_data, edges_data) where each is a list of rows.
		"""
		if schema_name is None:
			schema_name = self.graph_schema

		if graph_name is None or graph_name == "base":
			nodes_table = "graph_nodes"
			edges_table = "graph_edges"
		else:
			nodes_table = f"graph_nodes_{graph_name}"
			edges_table = f"graph_edges_{graph_name}"

		print(f"Retrieving Graph from tables {nodes_table} and {edges_table}")

		if all_columns:
			nodes_query = f'SELECT * FROM "{schema_name}"."{nodes_table}"'
			edges_query = f'SELECT * FROM "{schema_name}"."{edges_table}"'
		else:
			nodes_query = f'SELECT id, node, ST_AsGeoJSON(geom) AS geom FROM "{schema_name}"."{nodes_table}"'
			if weighted:
				edges_query = f'SELECT source, target, weight, ST_AsGeoJSON(geom) AS geom, adjusted_weight FROM "{schema_name}"."{edges_table}"'
			else:
				edges_query = f'SELECT source, target, weight, ST_AsGeoJSON(geom) AS geom FROM "{schema_name}"."{edges_table}"'

		with self.pg.connect() as conn:
			nodes_data = conn.execute(text(nodes_query)).fetchall()
			edges_data = conn.execute(text(edges_query)).fetchall()
		return nodes_data, edges_data

	def is_truly_undirected(self, schema_name=None, graph_name=None):
		"""
		Checks if the graph is truly undirected by verifying that for every edge (a,b),
		there exists a corresponding edge (b,a) with the same properties using SQLAlchemy.

		Returns:
			bool: True if the graph is truly undirected, False otherwise.
		"""
		if schema_name is None:
			schema_name = self.graph_schema

		if graph_name is None or graph_name == "base":
			edges_table = "graph_edges"
		else:
			edges_table = f"graph_edges_{graph_name}"

		try:
			# Assuming self.session is a SQLAlchemy session
			# or self.engine is a SQLAlchemy engine

			# Query to check for directed edges
			query = text(f"""
	        WITH edges AS (
	            SELECT source, target, weight
	            FROM "{schema_name}"."{edges_table}"
	        ),
	        directed_edges AS (
	            SELECT e1.source, e1.target
	            FROM edges e1
	            LEFT JOIN edges e2 ON e1.source = e2.target AND e1.target = e2.source
	            WHERE e2.source IS NULL
	            UNION
	            SELECT e1.source, e1.target
	            FROM edges e1
	            JOIN edges e2 ON e1.source = e2.target AND e1.target = e2.source
	            WHERE ABS(e1.weight - e2.weight) > 0.000001
	        )
	        SELECT COUNT(*) FROM directed_edges;
	        """)

			# Use existing session if available
			with self.pg.connect() as conn:
				result = conn.execute(query).scalar()

			# If count is 0, then the graph is truly undirected
			if result == 0:
				print(f"{datetime.now()} - Is undirected {graph_name} graph?")
			return result


		except Exception as e:
			# Use logger if available, otherwise fall back to print
			if hasattr(self, 'logger'):
				self.logger.error(f"Error checking if graph is undirected: {e}")
			else:
				print(f"Error checking if graph is undirected: {e}")
			return False

	def pg_export_graph_to_geopackage(self, graph_name=None, output_path=None, include_nodes=True, include_edges=True, weighted=False, all_columns: bool = False):
		"""
		Exports graph nodes and/or edges from PostGIS to a GeoPackage file.

		Parameters:
			graph_name (str, optional): Name of the graph to export. If None, uses the base graph.
			output_path (str, optional): Path where the GeoPackage will be saved.
										If None, uses 'graph_{graph_name}.gpkg'.
			include_nodes (bool): Whether to include nodes in the export (default: True)
			include_edges (bool): Whether to include edges in the export (default: True)
			weighted (bool, optional): If True, fetches the adjusted_weight for edges.
			all_columns (bool, optional): If True, retrieves all columns in the nodes and edges tables.

		Returns:
			str: Path to the created GeoPackage file.
		"""
		# Updated helper to parse geometry data from a column.
		def parse_geometry(x):
			from shapely import wkb, wkt
			try:
				if isinstance(x, dict):
					return shape(x)
				elif isinstance(x, str):
					# First, try loading as GeoJSON.
					try:
						geo = json.loads(x)
						return shape(geo)
					except json.JSONDecodeError:
						pass
					# Next, try to see if it is a hex-encoded WKB.
					# If the string contains only hex digits and has an even length, assume WKB hex.
					if all(c in "0123456789ABCDEFabcdef" for c in x.strip()) and len(x.strip()) % 2 == 0:
						try:
							return wkb.loads(x, hex=True)
						except Exception as wkb_err:
							print("Error parsing geometry as WKB hex:", wkb_err)
					# Fallback: try to load as WKT.
					try:
						return wkt.loads(x)
					except Exception as wkt_err:
						print("Error parsing geometry as WKT:", wkt_err)
				return x
			except Exception as e:
				print("Error parsing geometry:", e)
				return None

		if output_path is None:
			graph_suffix = graph_name if graph_name else "base"
			output_path = f"graph_{graph_suffix}.gpkg"

		# Retrieve nodes and edges using existing function.
		nodes_data, edges_data = self.pg_get_graph_nodes_edges(
			schema_name=self.graph_schema,
			graph_name=graph_name,
			weighted=weighted,
			all_columns=all_columns
		)

		if not nodes_data and not edges_data:
			print(f"No graph data found for graph_name: {graph_name}")
			return None

		# Process nodes if requested.
		if include_nodes and nodes_data:
			if all_columns:
				try:
					nodes_columns = nodes_data[0].keys()
				except AttributeError:
					nodes_columns = None
				nodes_df = pd.DataFrame(nodes_data, columns=nodes_columns)
				geo_col = "geom" if "geom" in nodes_df.columns else "geom_json"
			else:
				nodes_df = pd.DataFrame(nodes_data, columns=['id', 'node', 'geom_json'])
				geo_col = "geom_json"

			nodes_df['geometry'] = nodes_df[geo_col].apply(parse_geometry)
			if geo_col in nodes_df.columns and geo_col != 'geometry':
				nodes_df = nodes_df.drop(columns=[geo_col])
			nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry='geometry', crs="EPSG:4326")
			nodes_gdf.to_file(output_path, driver="GPKG", layer='nodes')
			print(f"Exported {len(nodes_gdf)} nodes to {output_path}, layer 'nodes'")

		# Process edges if requested.
		if include_edges and edges_data:
			if all_columns:
				try:
					edges_columns = edges_data[0].keys()
				except AttributeError:
					edges_columns = None
				edges_df = pd.DataFrame(edges_data, columns=edges_columns)
				geo_col = "geom" if "geom" in edges_df.columns else "geom_json"
			else:
				if weighted:
					edges_df = pd.DataFrame(edges_data, columns=['source', 'target', 'weight', 'geom_json', 'adjusted_weight'])
				else:
					edges_df = pd.DataFrame(edges_data, columns=['source', 'target', 'weight', 'geom_json'])
				geo_col = "geom_json"

			edges_df['geometry'] = edges_df[geo_col].apply(parse_geometry)
			if geo_col in edges_df.columns and geo_col != 'geometry':
				edges_df = edges_df.drop(columns=[geo_col])
			edges_gdf = gpd.GeoDataFrame(edges_df, geometry='geometry', crs="EPSG:4326")

			if include_nodes and nodes_data:
				edges_gdf.to_file(output_path, driver="GPKG", layer='edges', mode='a')
			else:
				edges_gdf.to_file(output_path, driver="GPKG", layer='edges')
			print(f"Exported {len(edges_gdf)} edges to {output_path}, layer 'edges'")

		return output_path

	def pg_load_graph(self, nodes_data, edges_data):
		"""
		Loads the graph from PostGIS by querying the "graph_nodes" and "graph_edges" tables.
		Converts node string representations into tuple keys and forms a NetworkX graph.
		"""
		G = nx.Graph()
		for row in nodes_data:
			try:
				# Convert node string (e.g. "(lon, lat)") into a tuple.
				node_key = ast.literal_eval(row[1])
				geom_json = json.loads(row[2])
				point = shape(geom_json)
				G.add_node(node_key, point=point)
			except Exception as e:
				print("Error processing node:", e)
		for row in edges_data:
			try:
				# row: (source, target, weight, geojson)
				source = ast.literal_eval(row[0])
				target = ast.literal_eval(row[1])
				weight = row[2]
				geom_json = json.loads(row[3])
				G.add_edge(source, target, weight=weight, geom=geom_json)
			except Exception as e:
				print("Error processing edge:", e)
		return G
