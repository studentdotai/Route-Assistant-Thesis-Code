class FineGraph(BaseGraph):
	"""
		FineGraph extends BaseGraph to provide additional capabilities
		for detailed routing and graph manipulation around specific areas.
		"""

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



	def pg_fine_grid(self, layer_name: str, enc_names: list, route_buffer, geom_column: str = "wkb_geometry",
					 save_to_db: bool = False, schema_name: str = None, table_name: str = "grid_fine"):
		"""
		Creates a fine grid by slicing and combining layer geometries that intersect with a buffer.
		Optimized to return only the combined geometry as GeoJSON for use with NetworkX and Plotly.
		Removes land areas from usage bands 4 and 5 from the resulting geometry.

		Parameters:
		  layer_name (str): The name of the table (in the ENC schema) containing layer geometries.
		  enc_names (list): List of ENC identifier strings to filter the features.
		  route_buffer (shapely.geometry.Polygon): A buffer polygon used for slicing geometries.
		  geom_column (str): Name of the geometry column in the table. (Default is "wkb_geometry".)
		  save_to_db (bool): Whether to save the grid to PostGIS (Default is False).
		  schema_name (str): Schema where the grid will be saved. If None, uses graph_schema.
		  table_name (str): Table name for the saved grid (Default is "fine_grid").

		Returns:
		  str: GeoJSON string representation of the combined sliced geometries with land areas removed.
		"""
		formated_names = self.pg._format_enc_names(enc_names)

		# Use the graph_schema as default if schema_name is None
		if schema_name is None:
			schema_name = self.graph_schema

		# Query to get combined geometry with land areas removed directly from PostgreSQL

		union_query = f"""
		   WITH combined_geometry AS (
			   SELECT ST_Union(
				   ST_Intersection({geom_column}, ST_GeomFromText(:wkt_buffer, 4326))
			   ) as geom
			   FROM "{self.enc_schema}"."{layer_name}"
			   WHERE dsid_dsnm = ANY(:enc_names)
				 AND ST_Intersects({geom_column}, ST_GeomFromText(:wkt_buffer, 4326))
		   ),
		   land_areas AS (
			   SELECT ST_Union(wkb_geometry) as geom
			   FROM "{self.enc_schema}"."lndare"
			   WHERE substring(dsid_dsnm from 3 for 1) IN ('4','5')
				 AND ST_Intersects(wkb_geometry, ST_GeomFromText(:wkt_buffer, 4326))
		   )
		   SELECT ST_AsGeoJSON(
			   CASE 
				   WHEN (SELECT geom FROM land_areas) IS NOT NULL 
				   THEN ST_Difference((SELECT geom FROM combined_geometry), (SELECT geom FROM land_areas))
				   ELSE (SELECT geom FROM combined_geometry)
			   END
		   ) as combined_geojson
		   """

		params = {"enc_names": formated_names, "wkt_buffer": route_buffer.wkt}

		with self.pg.connect() as conn:
			result = conn.execute(text(union_query), params).fetchone()

		geojson_string = None
		if result and result[0]:
			# Keep as string instead of parsing into a Python dict
			geojson_string = result[0]

			# Save to database if requested
			if save_to_db:
				create_schema_sql = f'CREATE SCHEMA IF NOT EXISTS "{schema_name}";'

				drop_table_sql = f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}"'

				create_table_sql = f"""
					CREATE TABLE IF NOT EXISTS "{schema_name}"."{table_name}" (
						id SERIAL PRIMARY KEY,
						grid GEOMETRY(Geometry,4326),
						created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
					);
				"""

				insert_sql = f"""
					INSERT INTO "{schema_name}"."{table_name}" (grid)
					VALUES (ST_GeomFromGeoJSON(:geojson));
				"""

				with self.pg.connect() as conn:
					conn.execute(text(create_schema_sql))
					conn.execute(text(drop_table_sql))
					conn.execute(text(create_table_sql))
					conn.execute(text(insert_sql), {"geojson": geojson_string})
					conn.commit()

				print(f"Fine grid saved to PostGIS in {schema_name}.{table_name}")

			return geojson_string
		else:
			# Return empty GeoJSON string if no geometries were found
			return '{"type": "GeometryCollection", "geometries": []}'


	def pg_filter_layer_by_buffer(self, layer_name: str, enc_names: list, route_buffer,
								  geom_column: str = "wkb_geometry"):
		"""
		Filters ENC layer geometries by ENC names and restricts the result to only those
		features that lie within the given route buffer polygon. In addition to returning
		a GeoDataFrame, this function also outputs the GeoJSON representation for use in
		Plotly visualizations and graph building.

		Parameters:
		  layer_name (str): The name of the table (in the ENC schema) containing layer geometries.
		  enc_names (list): List of ENC identifier strings to filter the features.
		  route_buffer (shapely.geometry.Polygon): A buffer polygon (e.g., route buffer) used for filtering.
		  geom_column (str): Name of the geometry column in the table. (Default is "wkb_geometry".)

		Returns:
		  tuple: (gdf, geojson_output)
			gdf (gpd.GeoDataFrame): GeoDataFrame of features filtered by ENC names and route buffer.
			geojson_output (str): GeoJSON string representing the filtered features.
		"""
		formated_names = self.pg._format_enc_names(enc_names)

		query = text(f"""
			   SELECT *, ST_AsText({geom_column}) as geom_wkt
			   FROM "{self.enc_schema}"."{layer_name}"
			   WHERE dsid_dsnm = ANY(:enc_names)
				 AND ST_Intersects({geom_column}, ST_GeomFromText(:wkt_buffer, 4326))
		   """)

		params = {"enc_names": formated_names, "wkt_buffer": route_buffer.wkt}

		with self.pg.connect() as conn:
			result = conn.execute(query, params)
			rows = result.fetchall()
			columns = result.keys()

		# Convert to DataFrame then GeoDataFrame
		df = pd.DataFrame(rows, columns=columns)
		if not df.empty and 'geom_wkt' in df.columns:
			# Convert the geometry column from WKT to shapely objects
			df["geometry"] = gpd.GeoSeries.from_wkt(df['geom_wkt'])

			# Convert Decimal types to float to avoid JSON serialization issues
			for col in df.select_dtypes(include=['object']).columns:
				try:
					if df[col].apply(lambda x: isinstance(x, Decimal)).any():
						df[col] = df[col].astype(float)
				except:
					pass

			gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
		else:
			# Create an empty GeoDataFrame with a geometry column to satisfy GeoPandas
			gdf = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")

		# Custom function to convert Decimal to float for JSON serialization
		def decimal_default(obj):
			if isinstance(obj, Decimal):
				return float(obj)
			raise TypeError

		# Convert the GeoDataFrame into GeoJSON with custom conversion for Decimal
		geojson_output = gdf.to_json(default=decimal_default)
		return gdf, geojson_output

	def pg_slice_layer_by_buffer(self, layer_name: str, enc_names: list, route_buffer,
								 geom_column: str = "wkb_geometry", merge_geometries: bool = False):
		"""
		Slices layer geometries by a buffer polygon using ST_Intersection.
		Returns only the portions of geometries that lie within the buffer.

		Parameters:
		  layer_name (str): The name of the table (in the ENC schema) containing layer geometries.
		  enc_names (list): List of ENC identifier strings to filter the features.
		  route_buffer (shapely.geometry.Polygon): A buffer polygon used for slicing geometries.
		  geom_column (str): Name of the geometry column in the table. (Default is "wkb_geometry".)
		  merge_geometries (bool): If True, returns an additional merged geometry for graph creation.

		Returns:
		  tuple: If merge_geometries is False: (gdf, geojson_output)
				 If merge_geometries is True: (gdf, geojson_output, merged_geometry)
			gdf (gpd.GeoDataFrame): GeoDataFrame of sliced features.
			geojson_output (str): GeoJSON string representing the sliced features.
			merged_geometry (shapely.geometry): Combined geometry of all sliced features (if merge_geometries is True)
		"""
		formated_names = self.pg._format_enc_names(enc_names)

		# Query that slices geometries using ST_Intersection
		base_query = f"""
			SELECT 
				*, 
				ST_AsText(
					ST_Intersection({geom_column}, ST_GeomFromText(:wkt_buffer, 4326))
				) as sliced_geom_wkt
			FROM "{self.enc_schema}"."{layer_name}"
			WHERE dsid_dsnm = ANY(:enc_names)
			  AND ST_Intersects({geom_column}, ST_GeomFromText(:wkt_buffer, 4326))
		"""

		# Add query for merged geometry if requested
		merged_geom = None
		if merge_geometries:
			merge_query = f"""
				SELECT ST_AsText(
					ST_Union(
						ST_Intersection({geom_column}, ST_GeomFromText(:wkt_buffer, 4326))
					)
				) as merged_geometry
				FROM "{self.enc_schema}"."{layer_name}"
				WHERE dsid_dsnm = ANY(:enc_names)
				  AND ST_Intersects({geom_column}, ST_GeomFromText(:wkt_buffer, 4326))
			"""

		params = {"enc_names": formated_names, "wkt_buffer": route_buffer.wkt}

		with self.pg.connect() as conn:
			result = conn.execute(text(base_query), params)
			rows = result.fetchall()
			columns = result.keys()

			# Get merged geometry if requested
			if merge_geometries:
				merge_result = conn.execute(text(merge_query), params).fetchone()
				if merge_result and merge_result[0]:
					merged_geom = wkt.loads(merge_result[0])

		# Convert to DataFrame then GeoDataFrame
		df = pd.DataFrame(rows, columns=columns)
		if not df.empty and 'sliced_geom_wkt' in df.columns:
			# Convert the sliced geometry column from WKT to shapely objects
			df["geometry"] = gpd.GeoSeries.from_wkt(df['sliced_geom_wkt'])

			# Convert Decimal types to float to avoid JSON serialization issues
			for col in df.select_dtypes(include=['object']).columns:
				try:
					if df[col].apply(lambda x: isinstance(x, Decimal)).any():
						df[col] = df[col].astype(float)
				except:
					pass

			gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
		else:
			# Create an empty GeoDataFrame with a geometry column to satisfy GeoPandas
			gdf = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")

		# Custom function to convert Decimal to float for JSON serialization
		def decimal_default(obj):
			if isinstance(obj, Decimal):
				return float(obj)
			raise TypeError

		# Convert the GeoDataFrame into GeoJSON with correct parameter name for custom handler
		geojson_output = gdf.to_json(default=decimal_default)

		if merge_geometries:
			return gdf, geojson_output, merged_geom
		else:
			return gdf, geojson_output

	def pg_get_graphs_list(self, schema_name=None):
		"""
		Lists all available graphs in the specified schema by identifying
		pairs of node and edge tables with matching suffixes.

		Parameters:
			schema_name (str, optional): Schema to search in. Defaults to self.graph_schema.

		Returns:
			list: List of dictionaries containing graph information with keys:
				  - graph_name: The suffix identifying the graph
				  - nodes_table: Full name of the nodes table
				  - edges_table: Full name of the edges table
		"""
		if schema_name is None:
			schema_name = self.graph_schema

		# Query to get all tables in the schema
		query = text("""
			SELECT table_name 
			FROM information_schema.tables 
			WHERE table_schema = :schema
			AND table_name LIKE 'graph_nodes%' OR table_name LIKE 'graph_edges%'
		""")

		available_graphs = []

		with self.pg.connect() as conn:
			tables = [row[0] for row in conn.execute(query, {"schema": schema_name}).fetchall()]

			# Find node tables
			node_tables = [t for t in tables if t.startswith('graph_nodes')]
			# Find edge tables
			edge_tables = [t for t in tables if t.startswith('graph_edges')]

			# Group by graph name suffix
			for node_table in node_tables:
				# Extract suffix (everything after "graph_nodes_")
				if len(node_table) > 11:  # Longer than just "graph_nodes"
					suffix = node_table[12:]
					matching_edge_table = f"graph_edges_{suffix}"

					if matching_edge_table in edge_tables:

						available_graphs.append(suffix)
				else:
					# Handle the case of default graph tables without suffix
					if "graph_edges" in edge_tables:
						available_graphs.append("base")

		# Print the available graphs in a formatted way
		if available_graphs:
			print(f"Available graphs in schema '{schema_name}':")
			for i, graph in enumerate(available_graphs, 1):
				print(f"{i}. Graph: {graph}")
		else:
			print(f"No graphs found in schema '{schema_name}'")

		return available_graphs

	def _enhance_graph_schema(self, graph_name="base", layer_tables=None):
		"""
		Enhances the graph edges table with additional columns for weight management,
		creating individual columns for each maritime feature layer.

		Parameters:
			graph_name (str): Name of the graph to enhance
			layer_tables (dict): Dictionary of layer names to include as weight columns
		"""
		edges_table = f"graph_edges_{graph_name}" if graph_name != "base" else "graph_edges"

		# Add base weight columns if they don't exist
		base_columns_sql = f"""
		DO $$ 
		BEGIN
			IF NOT EXISTS (
				SELECT 1 FROM information_schema.columns 
				WHERE table_schema = '{self.graph_schema}' 
				AND table_name = '{edges_table}' 
				AND column_name = 'base_weight'
			) THEN
				ALTER TABLE "{self.graph_schema}"."{edges_table}" 
				ADD COLUMN base_weight FLOAT,
				ADD COLUMN adjusted_weight FLOAT;

				-- Initialize base_weight from existing weight
				UPDATE "{self.graph_schema}"."{edges_table}"
				SET base_weight = weight,
					adjusted_weight = weight;
			END IF;
		END
		$$;
		"""

		# Generate SQL to add layer-specific weight columns
		layer_columns_sql = []
		if layer_tables:
			for layer_name in layer_tables:
				# Sanitize layer name for column name
				col_name = f"wt_{layer_name.lower()}"

				# Add column if it doesn't exist
				layer_sql = f"""
				DO $$
				BEGIN
					IF NOT EXISTS (
						SELECT 1 FROM information_schema.columns 
						WHERE table_schema = '{self.graph_schema}' 
						AND table_name = '{edges_table}' 
						AND column_name = '{col_name}'
					) THEN
						ALTER TABLE "{self.graph_schema}"."{edges_table}" 
						ADD COLUMN {col_name} FLOAT DEFAULT 1.0;
					END IF;
				END
				$$;
				"""
				layer_columns_sql.append(layer_sql)

		with self.pg.connect() as conn:
			# Add base columns
			conn.execute(text(base_columns_sql))

			# Add layer-specific columns
			if layer_columns_sql:
				for sql in layer_columns_sql:
					conn.execute(text(sql))

			conn.commit()

		print(f"Enhanced graph schema with weight management columns")

		# Return list of column names that were added (useful for apply_feature_weights)
		column_names = []
		if layer_tables:
			column_names = [f"wt_{layer_name.lower()}" for layer_name in layer_tables]
			print(f"Added columns: {', '.join(column_names)}")
		return column_names

	def pg_apply_feature_weights(self, graph_name="base", layer_tables=None, usage_bands=None, apply_to_weight=False):
		"""
		Calculates adjusted weights based on maritime features using individual columns
		for each layer. Original weights are preserved in base_weight column.

		Parameters:
			graph_name (str): Name of the graph to modify
			layer_tables (dict): Dictionary mapping layer names to their weight factors
			usage_bands (list): List of usage bands to include (e.g., ['1','3','4'])
			apply_to_weight (bool): If True, updates the main weight column with the calculated result. Defaults to False.
		"""
		if layer_tables is None:
			layer_tables = {
				'fairwy': {'attr': None, 'factor': 0.8},
				'tsslpt': {'attr': None, 'factor': 0.7},
				'depcnt': {'attr': 'valdco', 'values': {'5': 1.5, '10': 1.2, '20': 0.9}},
				'resare': {'attr': None, 'factor': 2.0},
				'obstrn': {'attr': None, 'factor': 5.0}
			}

		# Default usage bands (all) if not specified
		if usage_bands is None:
			usage_bands = ['1', '2', '3', '4', '5', '6']

		# Format the usage bands for SQL IN clause
		usage_bands_str = "'" + "','".join(usage_bands) + "'"

		# Ensure the enhanced schema exists with columns for each layer
		column_names = self._enhance_graph_schema(graph_name, layer_tables)

		# Determine table name
		edges_table = f"graph_edges_{graph_name}" if graph_name != "base" else "graph_edges"

		# Reset all weight columns to 1.0 (neutral factor)
		reset_weights_sql = ", ".join([f"{col} = 1.0" for col in column_names]) if column_names else ""
		if reset_weights_sql:
			reset_sql = f"""
			UPDATE "{self.graph_schema}"."{edges_table}"
			SET {reset_weights_sql};
			"""
			with self.pg.connect() as conn:
				conn.execute(text(reset_sql))
				conn.commit()

		# Generate SQL for each layer's weight tracking
		weight_calculations = []

		for layer_name, config in layer_tables.items():
			# Sanitize layer name for column usage
			col_name = f"wt_{layer_name.lower()}"

			if config.get('attr') and config.get('values'):
				# Attribute-based weights
				for attr_value, factor in config['values'].items():
					# Create column name for this specific attribute value
					attr_col_name = f"wt_{layer_name.lower()}_{attr_value.lower()}"

					# Make sure this column exists
					attr_col_sql = f"""
					DO $$
					BEGIN
						IF NOT EXISTS (
							SELECT 1 FROM information_schema.columns 
							WHERE table_schema = '{self.graph_schema}' 
							AND table_name = '{edges_table}' 
							AND column_name = '{attr_col_name}'
						) THEN
							ALTER TABLE "{self.graph_schema}"."{edges_table}" 
							ADD COLUMN {attr_col_name} FLOAT DEFAULT 1.0;
						END IF;
					END
					$$;
					"""


					# Get actual table name by stripping any suffix after underscore
					actual_layer = layer_name.split('_')[0]
					# SQL to apply weight factor if edge intersects with this feature
					# Add usage band filter to the WHERE clause
					weight_sql = f"""
					UPDATE "{self.graph_schema}"."{edges_table}" e
					SET {attr_col_name} = {factor}
					FROM "{self.enc_schema}"."{actual_layer}" l
					WHERE ST_Intersects(e.geom, l.wkb_geometry)
					AND l.{config['attr']} = '{attr_value}'
					AND substring(l.dsid_dsnm from 3 for 1) IN ({usage_bands_str});
					"""

					weight_calculations.append((attr_col_sql, weight_sql))
			else:
				# Fixed factor for entire layer
				factor = config.get('factor', 1.0)

				# SQL to apply weight factor if edge intersects with this feature
				# Add usage band filter to the WHERE clause
				weight_sql = f"""
				UPDATE "{self.graph_schema}"."{edges_table}" e
				SET {col_name} = {factor}
				FROM "{self.enc_schema}"."{layer_name}" l
				WHERE ST_Intersects(e.geom, l.wkb_geometry)
				AND substring(l.dsid_dsnm from 3 for 1) IN ({usage_bands_str});
				"""

				weight_calculations.append((None, weight_sql))

		# Calculate the final adjusted weight based on all individual weight columns
		# Get all weight columns from the table
		get_cols_sql = f"""
		SELECT column_name FROM information_schema.columns 
		WHERE table_schema = '{self.graph_schema}' 
		AND table_name = '{edges_table}'
		AND column_name LIKE 'wt_%';
		"""

		# Execute all SQL in transaction
		with self.pg.connect() as conn:
			try:
				conn.execute(text("BEGIN;"))

				# Ensure base_weight is populated
				conn.execute(text(f"""
					UPDATE "{self.graph_schema}"."{edges_table}" 
					SET base_weight = weight 
					WHERE base_weight IS NULL;
				"""))

				# Create any missing attribute-specific columns and apply weight factors
				for col_sql, weight_sql in weight_calculations:
					if col_sql:
						conn.execute(text(col_sql))
					conn.execute(text(weight_sql))

				# Get all weight columns
				weight_cols = [row[0] for row in conn.execute(text(get_cols_sql)).fetchall()]

				# Build the multiplication expression for all weight columns
				if weight_cols:
					weight_expr = " * ".join(["base_weight"] + weight_cols)

					# Update adjusted_weight
					update_sql = f"""
					UPDATE "{self.graph_schema}"."{edges_table}"
					SET adjusted_weight = {weight_expr}
					"""
					conn.execute(text(update_sql))

					# Optionally update main weight
					if apply_to_weight:
						weight_sql = f"""
						UPDATE "{self.graph_schema}"."{edges_table}"
						SET weight = adjusted_weight
						"""
						conn.execute(text(weight_sql))

				conn.execute(text("COMMIT;"))
				print(f"Successfully applied feature weights to graph '{graph_name}' using usage bands: {usage_bands}")
				return True

			except Exception as e:
				conn.execute(text("ROLLBACK;"))
				print(f"Error calculating weights: {str(e)}")
				return False

	# New function to load graph with specific weighting
	def pg_load_graph_with_weights(self, nodes_data, edges_data, use_adjusted_weights=True):
		"""
		Loads the graph from PostGIS data, with option to use original or adjusted weights

		Parameters:
			nodes_data, edges_data: Data from pg_get_graph_nodes_edges()
			use_adjusted_weights: If True, uses adjusted_weight instead of base_weight

		Returns:
			nx.Graph: NetworkX graph with selected weight values
		"""
		G = nx.Graph()

		for row in nodes_data:
			try:
				node_key = ast.literal_eval(row[1])
				geom_json = json.loads(row[2])
				point = shape(geom_json)
				G.add_node(node_key, point=point)
			except Exception as e:
				print("Error processing node:", e)

		for row in edges_data:
			try:
				source = ast.literal_eval(row[0])
				target = ast.literal_eval(row[1])

				# Extract weights - assume row structure includes adjusted_weight and base_weight
				if len(row) >= 4:  # Original format with just weight
					default_weight = row[2]
					geom_json = json.loads(row[3])

					# Simple case - just use the provided weight
					G.add_edge(source, target, weight=default_weight, geom=geom_json)
				else:  # Enhanced format with both weights
					base_weight = row[2]
					adjusted_weight = row[3] if row[3] is not None else base_weight
					factors = row[4]  # JSONB stored as string
					geom_json = json.loads(row[5])

					# Choose which weight to use
					weight_to_use = adjusted_weight if use_adjusted_weights else base_weight

					G.add_edge(source, target,
							   weight=weight_to_use,
							   base_weight=base_weight,
							   adjusted_weight=adjusted_weight,
							   weight_factors=factors,
							   geom=geom_json)
			except Exception as e:
				print("Error processing edge:", e)

		return G

	def pg_connect_nodes(self, source_id, target_id, custom_weight=None,
								graph_name=None):
		"""
		Creates a new edge between two existing nodes in the graph database using their primary key IDs.

		Parameters:
			source_id (int): Primary key ID of the source node
			target_id (int): Primary key ID of the target node
			custom_weight (float, optional): Custom weight for the edge. If None, calculated based on distance.
			nodes_table (str): Name of the nodes table (default "graph_nodes")
			edges_table (str): Name of the edges table (default "graph_edges")

		Returns:
			bool: True if edge creation was successful, False otherwise
		"""
		if graph_name is None or "base":
			nodes_table = "graph_nodes"
			edges_table = "graph_edges"
		else:
			try:
				nodes_table = f"graph_nodes_{graph_name}"
				edges_table = f"graph_edges_{graph_name}"
			except Exception as e:
				print("Error referencing Nodes and Edges tables. \nPlease provide a valid graph name.")


		# SQL to get node details based on their IDs
		node_query = f"""
		SELECT id, node, ST_AsText(geom) as geom_wkt
		FROM "{self.graph_schema}"."{nodes_table}"
		WHERE id IN (:source_id, :target_id)
		"""

		with self.pg.connect() as conn:
			try:
				# Get node information
				nodes_result = conn.execute(text(node_query),
											{"source_id": source_id, "target_id": target_id})
				nodes = nodes_result.fetchall()

				if len(nodes) != 2:
					print(f"Error: One or both nodes (IDs: {source_id}, {target_id}) not found.")
					return False

				# Map node details
				node_map = {row[0]: {"node_str": row[1], "geom": row[2]} for row in nodes}

				# Check if edge already exists
				check_edge_sql = f"""
				SELECT COUNT(*) FROM "{self.graph_schema}"."{edges_table}"
				WHERE (source_id = :source_id AND target_id = :target_id)
				OR (source_id = :target_id AND target_id = :source_id)
				"""

				edge_exists = conn.execute(text(check_edge_sql),
										   {"source_id": source_id, "target_id": target_id}).scalar()

				if edge_exists > 0:
					print(f"Edge between nodes {source_id} and {target_id} already exists.")
					return False

				# Calculate weight based on distance if not provided
				if custom_weight is None:
					weight_sql = f"""
					SELECT ST_Distance(
						ST_GeomFromText(:source_geom, 4326),
						ST_GeomFromText(:target_geom, 4326)
					) as weight
					"""
					weight_result = conn.execute(text(weight_sql), {
						"source_geom": node_map[source_id]["geom"],
						"target_geom": node_map[target_id]["geom"]
					}).scalar()
					weight = weight_result
				else:
					weight = custom_weight

				# Create the edge
				insert_edge_sql = f"""
				INSERT INTO "{self.graph_schema}"."{edges_table}" 
				(source, target, source_id, target_id, weight, geom)
				VALUES (
					:source_node_str, 
					:target_node_str, 
					:source_id, 
					:target_id, 
					:weight, 
					ST_MakeLine(
						(SELECT geom FROM "{self.graph_schema}"."{nodes_table}" WHERE id = :source_id),
						(SELECT geom FROM "{self.graph_schema}"."{nodes_table}" WHERE id = :target_id)
					)
				)
				"""

				conn.execute(text(insert_edge_sql), {
					"source_node_str": node_map[source_id]["node_str"],
					"target_node_str": node_map[target_id]["node_str"],
					"source_id": source_id,
					"target_id": target_id,
					"weight": weight
				})

				conn.commit()
				print(f"Successfully created edge between nodes {source_id} and {target_id} with weight {weight}")
				return True

			except Exception as e:
				print(f"Error creating edge: {str(e)}")
				return False

	@staticmethod
	def verify_graph(G, nodes_gdf, edges_gdf):
		"""
		Verify that the graph has the correct number of nodes and edges
		and consists of a single connected component.

		Parameters:
		-----------
		G : NetworkX Graph
			The graph to verify
		nodes_gdf : GeoDataFrame
			Original nodes GeoDataFrame
		edges_gdf : GeoDataFrame
			Original edges GeoDataFrame

		Returns:
		--------
		is_valid : bool
			True if the graph is valid, False otherwise
		"""
		# Check node count
		expected_nodes = len(nodes_gdf)
		actual_nodes = G.number_of_nodes()
		nodes_match = expected_nodes == actual_nodes

		# Check edge count
		expected_edges = len(edges_gdf)
		actual_edges = G.number_of_edges()
		edges_match = expected_edges == actual_edges

		# Check connectivity
		is_connected = nx.is_connected(G)

		# Print verification results
		print(f"Node count: Expected {expected_nodes}, Actual {actual_nodes}, Match: {nodes_match}")
		print(f"Edge count: Expected {expected_edges}, Actual {actual_edges}, Match: {edges_match}")
		print(f"Graph is connected: {is_connected}")

		if not is_connected:
			# Find connected components
			components = list(nx.connected_components(G))
			print(f"Number of connected components: {len(components)}")
			print(f"Sizes of components: {[len(c) for c in components]}")

		return nodes_match and edges_match and is_connected

	def pg_copy_table(self, source_table, target_table, schema_name):
		"""
		Creates a copy of a PostgreSQL table using SQLAlchemy.

		Args:
			source_table (str): Name of the source table
			target_table (str): Name of the target table
			schema_name (str): Schema name

		Returns:
			bool: True if successful, False otherwise
		"""
		try:
			# Clean up table names - remove any spaces
			source_table = source_table.strip()
			target_table = target_table.strip()

			# Check if target table already exists
			check_sql = text(f"""
	            SELECT EXISTS (
	                SELECT FROM information_schema.tables 
	                WHERE table_schema = :schema 
	                AND table_name = :table
	            )
	        """)

			# Drop table if it exists
			drop_sql = text(f'DROP TABLE IF EXISTS "{schema_name}"."{target_table}"')

			# Create new table as copy
			copy_sql = text(f'CREATE TABLE "{schema_name}"."{target_table}" AS TABLE "{schema_name}"."{source_table}"')

			with self.pg.connect() as conn:
				# Check if table exists
				exists = conn.execute(check_sql, {"schema": schema_name, "table": target_table}).scalar()

				# Start transaction
				conn.execute(text("BEGIN;"))

				# Drop if exists
				if exists:
					conn.execute(drop_sql)

				# Create copy
				conn.execute(copy_sql)

				# Commit transaction
				conn.execute(text("COMMIT;"))

			print(f"Table {schema_name}.{target_table} created as a copy of {schema_name}.{source_table}")
			return True

		except Exception as e:
			print(f"Error copying table: {str(e)}")
			with self.pg.connect() as conn:
				conn.execute(text("ROLLBACK;"))
			return False
