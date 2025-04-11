class PostGIS:
	def __init__(self):
		self.data = Data()
		self.noaa_db = NOAA_DB()
		self.data.s57_attributes_df()
		self.data.s57_objects_df()
		self.data.s57_properties_df()

		self.engine = None
		self.session = None
		self.connection = None

		load_dotenv("SECRET.env")

		self.DB_CONFIG = {
			'host': os.getenv('DB_HOST'),
			'database': os.getenv('DB_NAME'),
			'user': os.getenv('DB_USER'),
			'password': os.getenv('DB_PASSWORD'),
			'port': os.getenv('DB_PORT')
		}


	def connect(self):
		"""Establish connection to PostGIS database using SQLAlchemy"""
		try:
			connection_string = f"postgresql://{self.DB_CONFIG['user']}:{self.DB_CONFIG['password']}@{self.DB_CONFIG['host']}:{self.DB_CONFIG['port']}/{self.DB_CONFIG['database']}"
			self.engine = create_engine(connection_string)
			self.session = sessionmaker(bind=self.engine)
			self.connection = self.engine.connect()
			return self.connection
		except Exception as e:
			print(f"Connection error: {e}")
			return False

	def is_connection_alive(self):
		try:
			result = self.connection.execute("SELECT 1;")
			return result.fetchone() is not None
		except Exception as e:
			print("Connection check failed:", e)
			return False

	def connection_test(self):
		if self.connection is None or not self.is_connection_alive():
			try:
				self.connection = self.connect()  # or however you reconnect
			except Exception as e:
				print("Reconnection attempt failed:", e)
				return False
		return True

	def _tables_exist(self, schema = 'public'):
		"""Check if tables exist in the database"""
		try:
			inspector = inspect(self.engine)
			table_names = inspector.get_table_names(schema=schema)
			print(f"Tables in the database: {table_names}")
			return True
		except Exception as e:
			print(f"Error checking tables: {e}")
			return False

	def _format_enc_names(self, enc_names: Union[str, List[str]]) -> List[str]:
		"""
		Formats ENC names to S-57 standard with .000 extension.

		Args:
			enc_names: Single ENC name or list of ENC names

		Returns:
			List[str]: ENC names formatted as NAME.000

		Examples:
			format_enc_names('US5CA51M') -> ['US5CA51M.000']
			format_enc_names(['US5CA51M', 'US5CA52M.000']) -> ['US5CA51M.000', 'US5CA52M.000']
		"""
		# Convert single string to list
		if isinstance(enc_names, str):
			enc_names = [enc_names]

		formatted_names = []
		for name in enc_names:
			# Remove any existing extensions (.000, .gpkg)
			base_name = name.split('.')[0]
			formatted_names.append(f"{base_name}.000")
		return formatted_names

	def enc_db_summary(self, schema_name = 'public', detailed: bool = False, show_outdated: bool = False, noaa_data: bool = False):
		"""
		Retrieves comprehensive ENC summary information from PostGIS database.

	    Args:
	    	schema_name (str): Database schema name. Defaults to 'public'.
	        detailed (bool): If True, provides detailed information unified column Names
	        show_outdated (bool): If True, includes outdated ENC status.

	    Returns:
            pd.DataFrame: A DataFrame summarizing ENC information.
		"""
		if not self.connect():
			print("Connection not established. Please run connect() first.")
			return

		table_name = 'dsid'
		columns = ['dsid_dsnm', 'dsid_edtn', 'dsid_updn']

		raw_sql = f"""
		SELECT {', '.join(columns)}
		FROM "{schema_name}".{table_name};
		"""

		# Convert query result to GeoPandas DataFrame
		df = pd.read_sql(raw_sql, self.engine)
		cleaned_df = self.data.clean_enc_names_column(df, 'dsid_dsnm')

		noaa_df = self.noaa_db.get_dataframe()
		if show_outdated:
			# Check for outdated ENCs against NOAA database
			for idx, row in cleaned_df.iterrows():
				db_entry = noaa_df[noaa_df['ENC_Name'] == row['dsid_dsnm']]
				if not db_entry.empty:
					db_edition = db_entry['Edition'].iloc[0]
					db_update = db_entry['Update'].iloc[0]
					cleaned_df.loc[idx, 'OUTDATED'] = (
							(row['dsid_edtn'] < db_edition) or
							(row['dsid_edtn'] == db_edition and row['dsid_updn'] < db_update)
					)
					if noaa_data:
						get_string = f"Ed: {db_edition}, Upd: {db_update}, Date: {db_entry['Update_Application_Date'].iloc[0]}"
						cleaned_df.loc[idx, 'NOAA_DATA'] = get_string
				else:
					cleaned_df.loc[idx, 'OUTDATED'] = False


		if detailed:
			cleaned_df = cleaned_df.rename(columns={'dsid_dsnm': 'ENC_NAME', 'dsid_edtn': 'ENC_EDITION', 'dsid_updn': 'ENC_UPDATE'})


		return cleaned_df



	def layers_summary(self, schema_name: str = 'public', cleaned: bool = False):
		"""
		Retrieves layer information from the specified database schema.

		Args:
			schema_name (str): The name of the schema to inspect. Defaults to 'public'.
			cleaned (bool): If True, filters out layers with zero entries and missing names.

		Returns:
			pd.DataFrame: A DataFrame containing layer names, acronyms, and entry counts.
		"""
		if not self.connect():
			print("Connection not established. Please run connect() first.")
			return

		# Get table names from the database
		inspector = inspect(self.engine)
		table_names = inspector.get_table_names(schema=schema_name)

		data = Data()
		layers = []
		acronyms = []
		entries = []
		for table in table_names:
			count_query = text(f""" SELECT COUNT(*) FROM "{schema_name}"."{table}" """)

			full_name = data.s57_objects_convert(table.upper())
			with self.engine.connect() as connection:
				result = connection.execute(count_query)
				count = result.scalar()
			layers.append(full_name)
			acronyms.append(table)
			entries.append(count)

		df = pd.DataFrame({'Layer name': layers,
							 'Acronym': acronyms,
							 'Entries': entries})

		if cleaned:
			df = df[(df['Entries'] > 0) & (df['Layer name'].notna())]
			return df
		else:
			return df

	def sort_enc(self, sort_input: str, sort_by: str = "usage band",  schema_name = "public" , output: str = "list", ascending = True):
		"""
		Sort and filter Electronic Navigational Charts (ENC) data from PostGIS database.

		Usage Bands:
		1: Overview
		2: General
		3: Coastal
		4: Approach
		5: Harbour
		6: Berthing

		Args:
		sort_by (str):
			Attribute to sort by. Options:
				- 'usage band'
				- 'code'
				- 'number'
		sort_input (str):
			Filter value corresponding to `sort_by`.
		output (str, optional):
			Format of the returned data. Choices:
				- 'list' (default)
				- 'dataframe'
		ascending (bool, optional):
			Sort order. `True` for ascending (default), `False` for descending.

		Returns:
			Union[List[str], pd.DataFrame, Dict[str, List[str]]]:
            - List of sorted `dsid_dsnm` values if `output` is 'list'
            - DataFrame sorted by `dsid_dsnm` if `output` is 'dataframe'
            - Dictionary with usage bands as keys if `output` is 'dict'
            - `None` if database connection fails

		 Examples:

		# Get all Usage Band 1 ENCs as DataFrame
		sort_enc('Usage Band', '1', True, 'dataframe')

		# Get US ENCs as list
		sort_enc('Code', 'US', True, 'list')

		# Search specific ENC number or part on the number
		sort_enc('Number', '11', True, 'dataframe')
		"""

		if not self.connect():
			print("Connection not established. Please run connect() first.")
			return


		table_name = 'dsid'
		columns = 'dsid_dsnm'

		pattern =   {'pattern': f'__{sort_input}%'} if sort_by.lower() == 'usage band' else \
					{'pattern': f'{sort_input}%'}   if sort_by.lower() == 'code' else \
					{'pattern': f'%{sort_input}%'}  if sort_by.lower() == 'number' else \
					{'pattern': '%'}


		raw_sql = f"""
		SELECT {columns}
		FROM "{schema_name}"."{table_name}"
		WHERE dsid_dsnm LIKE %(pattern)s
		"""

		df  = pd.read_sql(raw_sql, self.engine, params=pattern)
		sorted_df = df.sort_values(by='dsid_dsnm', ascending=ascending)
		return sorted_df['dsid_dsnm'].tolist() if output.lower() == 'list' else sorted_df

	def enc_bbox(self, enc_names: list[str] = None, schema_name: str = 'public'):
		"""
		Retrieves bounding geometries for specified ENC names from the database.

		Parameters:
		- enc_names (list): A list of ENC names to query.
		- schema_name (str): The name of the schema to query.
		Returns:
		- GeoDataFrame: A GeoDataFrame containing the geometries sorted by 'dsid_dsnm'.
		"""

		if not self.connect():
			print("Connection not established. Please run connect() first.")
			return
		if enc_names is None:
			print("Please provide a list of ENC names.")
			return
		# Format names to S-57 standard
		formatted_names = self._format_enc_names(enc_names)


		table_name = 'm_covr'
		columns = ['dsid_dsnm', 'wkb_geometry']

		# Verify the table exists and has data:
		verify_sql = f"""
						SELECT COUNT(*)
						FROM "{schema_name}"."{table_name}"
						"""
		verify_table =  pd.read_sql(verify_sql, self.engine )

		if not verify_table.empty:
			print(f"Table has: {verify_table['count'][0]} entries")
		else:
			print (f"Table {schema_name}.{table_name} is empty")

		# Generate the SQL query with placeholders

		placeholders = ', '.join(['%s'] * len(enc_names))
		raw_sql = f"""
				SELECT {', '.join(columns)}
				FROM "{schema_name}"."{table_name}"
				WHERE dsid_dsnm IN ({placeholders})
				"""
		# Convert enc_names directly to tuple
		pattern = tuple(formatted_names)

		gdf = gpd.read_postgis(raw_sql, self.engine, params=pattern, geom_col='wkb_geometry')
		gdf = self.data.clean_enc_names_column(gdf, 'dsid_dsnm')


		return gdf.sort_values(by='dsid_dsnm')


	def get_layer(self, schema_name = 'public', layer_name: str = "dsid", filter_by_enc: list[str] = "ALL", progress_callback = None):

		if not self.connect():
			print("Connection not established. Please run connect() first.")
			return

		if filter_by_enc != "ALL":
			filter_by_enc = self._format_enc_names(filter_by_enc)
			placeholders = ', '.join(['%s'] * len(filter_by_enc))
			raw_sql = f"""
					SELECT *
					FROM "{schema_name}"."{layer_name}"
					WHERE dsid_dsnm IN ({placeholders})
					"""
			pattern = tuple(filter_by_enc)
			gdf = gpd.read_postgis(raw_sql, self.engine, params=pattern,  geom_col='wkb_geometry')

		else:
			raw_sql = f"""
					SELECT *
					FROM "{schema_name}"."{layer_name}"
					"""
			gdf = gpd.read_postgis(raw_sql, self.engine, geom_col='wkb_geometry')
		print(len(gdf))

		return gdf.sort_values(by='dsid_dsnm')
