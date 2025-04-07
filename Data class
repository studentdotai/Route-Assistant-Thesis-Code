class Data:
	def __init__(self):
		self.ports_msi_shp = 'GIS_files/World Port Index_2019Shapefile/WPI.shp'
		self.ports_msi_acronyms = 'GIS_files/WorldPortIndex_2019.csv'
		self.land_shp = "GIS_files/ne_10m_land/ne_10m_land.shp"
		self.grid_shp = "GIS_files/ne_10m_graticules_1/ne_10m_graticules_1.shp"
		self.coast_10_shp = "GIS_files/ne_10m_coastline/ne_10m_coastline.shp"
		self.coast_110_shp = "GIS_files/ne_110m_coastline/ne_110m_coastline.shp"
		self.ocean_shp = "GIS_files/ne_110m_ocean"

		self.enc_folder = "GIS_files/CGD11_ENCs"
		self.all_enc_folder = "GIS_export/ALL_ENCs"

		self.input_folder = 'GIS_files/'
		self.output_folder = 'GIS_export/'

		self.s57_attributes_csv = 'GIS_files/s57attributes.csv'
		self.s57_objects_csv = 'GIS_files/s57objectclasses.csv'
		self.s57_expectedInput_csv = 'GIS_files/s57expectedinput.csv'

	@staticmethod
	def parse_string(s: str) -> str:
		"""
	    Extracts string content after colon if present, removing parentheses.

	    Args:
	        s (str): Input string, e.g., '(1:6)' or '(2:3,6)'
	    Returns:
	        str: Content after colon with parentheses removed
	    """
		if ':' in s:
			s = s.strip('()')
			return s.split(':')[1].strip()
		return s

	def clean_enc_name(enc_name: str) -> str:
		"""
		Cleans ENC filename by removing common file extensions.
		Args:
			enc_name: ENC filename to clean
		Returns:
			str: Cleaned ENC name without extensions
		Examples:
			clean_enc_name("US5VA51M.000") -> "US5VA51M"
			clean_enc_name("US5VA51M.gpkg") -> "US5VA51M"
			clean_enc_name("US5VA51M") -> "US5VA51M"
		"""
		# Split on first period and take base name
		return enc_name.split('.')[0]

	def clean_enc_names_column(self, df, column_name: str) -> pd.DataFrame:
		"""
		Cleans ENC names in specified DataFrame column by removing file extensions.
		Args:
			df: Input DataFrame containing ENC names
			column_name: Name of column containing ENC filenames
		Returns:
			pd.DataFrame: DataFrame with cleaned ENC names
		Examples:
			# Clean 'ENC_NAME' column
			df = clean_enc_names_column(df, 'ENC_NAME')
		"""
		df = df.copy()
		df[column_name] = df[column_name].str.split('.').str[0]
		return df

	def s57_attributes_df(self):
		"""
		Read a S-57 Attribute CSV file and convert it into a DataFrame.

		:param csv_file: Path to the CSV file
		:return: pandas DataFrame or None if file doesn't exist
		"""
		if not self.s57_attributes_csv or not os.path.exists(self.s57_attributes_csv):
			raise FileNotFoundError(f"CSV file not found: {self.s57_attributes_csv}")

		try:
			df = pd.read_csv(self.s57_attributes_csv)
			df.set_index('Code', inplace=True)
			return df
		except Exception as e:
			print(f"Error reading CSV file: {e}")
			return None

	def s57_objects_df(self):
		"""
		Read a S-57 Attribute CSV file and convert it into a DataFrame.

		:param csv_file: Path to the CSV file
		:return: pandas DataFrame or None if file doesn't exist
		"""
		if not self.s57_objects_csv or not os.path.exists(self.s57_objects_csv):
			raise FileNotFoundError(f"CSV file not found: {self.s57_objects_csv}")

		try:
			df = pd.read_csv(self.s57_objects_csv)
			df.set_index('Code', inplace=True)
			return df
		except Exception as e:
			print(f"Error reading CSV file: {e}")
			return None

	def s57_properties_df(self):
		"""
		Read a S-57 Attribute CSV file and convert it into a DataFrame.

		:param csv_file: Path to the CSV file
		:return: pandas DataFrame or None if file doesn't exist
		"""
		if not self.s57_expectedInput_csv or not os.path.exists(self.s57_expectedInput_csv):
			raise FileNotFoundError(f"CSV file not found: {self.s57_expectedInput_csv}")

		attr_df = self.s57_attributes_df()

		try:
			df = pd.read_csv(self.s57_expectedInput_csv)
			df.set_index('Code', inplace=True)

			# Filter attr_df to only include codes present in df
			attr_df = attr_df[attr_df.index.isin(df.index)]

			prop_df = pd.merge(df, attr_df, on='Code', how='outer')
			prop_df.insert(1, 'Acronym', prop_df.pop('Acronym'))
			prop_df.insert(2, 'Attribute', prop_df.pop('Attribute'))
			# Clean all ID rows with value NaN
			prop_df = prop_df.dropna(subset=['ID'])
			return prop_df
		except Exception as e:
			print(f"Error reading CSV file: {e}")
			return None

	def s57_attributes_convert(self, acronym_str):
		"""
		Convert an acronym string into a attribute string.
		Note: make sure acronym_str is Uppercase. Add .upprer() before passing string to this function
		:param acronym_str: String containing acronyms separated by semicolons.
		:return: String of attributes or None if acronyms not found
		"""

		attribute_df = self.s57_attributes_df()
		if acronym_str in attribute_df['Acronym'].values:
			return attribute_df.loc[attribute_df['Acronym'] == acronym_str, 'Attribute'].iloc[0]
		return None

	def s57_objects_convert(self, acronym_str):
		"""
		Convert an acronym string into a attribute string.
		Note: make sure acronym_str is Uppercase. Add .upprer() before passing string to this function
		:param acronym_str: String containing acronyms separated by semicolons
		:return: String of attributes or None if acronyms not found
		"""
		object_df = self.s57_objects_df()
		if acronym_str in object_df['Acronym'].values:
			return object_df.loc[object_df['Acronym'] == acronym_str, 'ObjectClass'].iloc[0]
		return None

	def s57_properties_convert(self, acronym_str: str, property_value: Any, prop_mixed: bool = False, debug = False) -> Union[ str, List[str] ]:
		"""
	    Convert an S-57 layer property value to meaningful names.

	    Args:
	        acronym_str (str): S-57 property acronym (e.g., 'NATSUR').
	        property_value (Any): Value or ID associated with the property.
	        prop_mixed (bool, optional):
	            If True, returns "Name (Code)". If False, returns only "Name". Defaults to False.
	        debug (bool, optional):
	            If True, prints debug information. Defaults to False.

	    Returns:
	        Union[str, List[str]]:
	            - A single string if `property_value` corresponds to one entry.
	            - A list of strings if multiple values are processed.
	            - Original `property_value` if conversion isn't applicable.

	    Examples:
	        data.s57_properties_convert('NATSUR', '(1:4,1)')
	        ['sand', 'mud']

	        data.s57_properties_convert('SOUND', '4,6')
	        ["Sound Level 4 (4)", "Sound Level 6 (6)"]
	    """
		prop_df = self.s57_properties_df()

		# Early validation
		if property_value is None or pd.isna(property_value):
			return None

		if isinstance(property_value, (int, float)):
			if property_value == -2147483648:
				return None
			if property_value < 0:
				return property_value



		# Determin if Property neme is in the propertytable
		if acronym_str in prop_df['Acronym'].values and property_value is not None:
			if debug:
				print(f"Acronym found: {acronym_str} ")
			attrType = prop_df.loc[prop_df['Acronym'] == acronym_str, 'Attribute'].head(1).iloc[0]
			# Check if property is Sting and need to be parsed to check IDs with property_df
			if isinstance(property_value, str):
				if debug:
					print(f"String value found: {property_value}")
				# Filters Free Text Strings
				if attrType == ("S"):
					return property_value
				# If property has multiple values they separated by ":", that needs to be cleaned
				if ":" in property_value:
					parsed_properties = []
					property_value = self.parse_string(property_value)
					if debug:
						print(f"Parsed string value: {property_value} ({type(property_value)})")
					if all(num.isdigit() for num in property_value.split(',')):
						numbers = [int(x) for x in property_value.split(',')]
						for number in numbers:
							matching_rows = prop_df.loc[(prop_df['Acronym'] == acronym_str) & (prop_df['ID'] == number), 'Meaning'].iloc[0]
							print(f"Matching rows: {matching_rows}")
							if prop_mixed:
								matching_rows = f"{matching_rows} ({number})"


							parsed_properties.append(matching_rows)
						return parsed_properties

					else:
						return property_value
					# Handle simple comma-separated values without colons
				elif "," in property_value and all(num.strip().isdigit() for num in property_value.split(',')):
					parsed_properties = []
					numbers = [int(x.strip()) for x in property_value.split(',')]
					for number in numbers:
						try:
							matching_rows = prop_df.loc[
								(prop_df['Acronym'] == acronym_str) & (prop_df['ID'] == number), 'Meaning'].iloc[0]
							if debug:
								print(f"Matching rows: {matching_rows}")
							if prop_mixed:
								matching_rows = f"{matching_rows} ({number})"
							parsed_properties.append(matching_rows)
						except IndexError:
							if debug:
								print(f"No matching meaning found for {acronym_str} with ID {number}")
							parsed_properties.append(str(number))
					return parsed_properties
					# Handle single numeric strings (e.g., "1")
				elif property_value.isdigit():
					try:
						number = int(property_value)
						matching_rows = \
						prop_df.loc[(prop_df['Acronym'] == acronym_str) & (prop_df['ID'] == number), 'Meaning'].iloc[0]
						if debug:
							print(f"Matching rows: {matching_rows}")
						if prop_mixed:
							matching_rows = f"{matching_rows} ({number})"
						return matching_rows
					except IndexError:
						if debug:
							print(f"No matching meaning found for {acronym_str} with ID {property_value}")
						return property_value

			if isinstance(property_value, list):
				if debug:
					print(f"Processing list: {property_value}")

				parsed_properties = []

				# Handle nested lists like [['1'], ['2']]
				if all(isinstance(x, list) for x in property_value):
					if debug:
						print(f"Processing list as list of lists")
					for item in property_value:
						if item and isinstance(item[0], str) and item[0].isdigit():
							number = int(item[0])
							try:
								matching_rows = prop_df.loc[
									(prop_df['Acronym'].astype(str) == acronym_str) &
									(prop_df['ID'].astype(int) == number), 'Meaning']

								if not matching_rows.empty:
									meaning = matching_rows.iloc[0]
									if prop_mixed:
										meaning = f"{meaning} ({number})"
									parsed_properties.append(meaning)
								else:
									parsed_properties.append(str(number))
							except (IndexError, KeyError):
								parsed_properties.append(str(number))

				# Handle simple lists like ['1', '3', '5'] or ['3'] or ['31,33']
				else:
					for item in property_value:
						if isinstance(item, str):
							# Handle comma-separated values within the string
							if ',' in item and all(num.strip().isdigit() for num in item.split(',')):
								# Split the comma-separated string and process each number
								numbers = [int(x.strip()) for x in item.split(',')]
								for number in numbers:
									try:
										matching_rows = prop_df.loc[
											(prop_df['Acronym'].astype(str) == acronym_str) &
											(prop_df['ID'].astype(int) == number), 'Meaning']

										if not matching_rows.empty:
											meaning = matching_rows.iloc[0]
											if prop_mixed:
												meaning = f"{meaning} ({number}) "
											parsed_properties.append(meaning)
										else:
											parsed_properties.append(str(number))
									except (IndexError, KeyError):
										parsed_properties.append(str(number))
							# Handle single digits
							elif item.isdigit():
								number = int(item)
								try:
									matching_rows = prop_df.loc[
										(prop_df['Acronym'].astype(str) == acronym_str) &
										(prop_df['ID'].astype(int) == number), 'Meaning']

									if not matching_rows.empty:
										meaning = matching_rows.iloc[0]
										if prop_mixed:
											meaning = f"{meaning} ({number})"
										parsed_properties.append(meaning)
									else:
										parsed_properties.append(str(number))
								except (IndexError, KeyError):
									parsed_properties.append(str(number))
				if debug:
					print(f"Return: {parsed_properties}")
				return parsed_properties

			# Filters integer values (e.g. "SCAMIN", "Compilation Scale", "Soundingdistance") that has no String property
			if attrType == "I":
				return property_value
			else:
				matching_rows = prop_df.loc[(prop_df['Acronym'] == acronym_str) & (prop_df['ID'] == property_value), 'Meaning'].iloc[0]
				print(f"Matching rows: {matching_rows}")
				if prop_mixed:
					matching_rows = f"{matching_rows} ({property_value})"
				return matching_rows

		elif isinstance(property_value, str) and ":" in property_value:
			property_values = self.parse_string(property_value)
		else:
			return property_value
