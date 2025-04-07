class NOAA_DB:
	"""
	A class that scrapes and manages NOAA Electronic Navigational Charts (ENC) data from the official website.

	Attributes:
		url (str): The URL of the NOAA ENC website (https://www.charts.noaa.gov/ENCs/ENCsIndv.shtml)
		df (pandas.DataFrame): DataFrame containing the scraped ENC data
		session (requests.Session): HTTP session for making requests

	Methods:
		get_dataframe():
			Returns the scraped data as a pandas DataFrame.
			Returns: pandas.DataFrame containing ENC information

		save_to_csv(filename="ENC_DB.csv"):
			Saves the DataFrame to a CSV file.
			Args:
				filename (str): Name of output CSV file

	Usage Example:
		noaa_db = NOAA_DB()
		enc_data = noaa_db.get_dataframe()
		noaa_db.save_to_csv("my_enc_data.csv")
	"""

	def __init__(self):
		self.url = "https://www.charts.noaa.gov/ENCs/ENCsIndv.shtml" # URL of the NOAA ENC website
		self.df = None
		self.session = requests.Session()

	@staticmethod
	def create_dataframe(headers, rows):
		"""Create a pandas DataFrame from the parsed table data."""
		df = pd.DataFrame(rows, columns=headers)
		df = df[df['#'] != '#']  # Remove rows which duplicate header names
		df = df.set_index('#')  # Set index to the "#" column
		df.columns = (df.columns
		              .str.replace('\xa0', ' ')
		              .str.strip()
		              .str.replace(' ', '_'))
		return df

	@staticmethod
	def parse_table(soup):
		"""Extract and parse the table data from the HTML content."""
		table = soup.find('table')
		if not table:
			raise ValueError("No table found in the HTML content")

		inner_tbody = table.find('tr').find('td').find_all('tr')  # .find_all('td')

		columns = []
		for row in inner_tbody:
			row_list = row.find_all('td')
			for i in range(9):
				column = row_list[i].text
				columns.append(column)

		# Separate headers and data rows
		headers = columns[:9]
		rows = [columns[i:i + 9] for i in range(0, len(columns), 9)]
		return headers, rows

	def _get_data(self):
		"""Fetch the HTML content from the NOAA ENC website."""
		try:
			response = self.session.get(self.url, timeout=10)
			response.raise_for_status()
			return BeautifulSoup(response.content, 'html.parser')
		except requests.RequestException as e:
			raise ConnectionError(f"Failed to fetch data from {self.url}: {str(e)}")

	def _scrape_enc_data(self):
		"""Main method to scrape the ENC data and create a DataFrame."""
		try:
			soup = self._get_data()
			headers, rows = self.parse_table(soup)
			self.df = self.create_dataframe(headers, rows)
		except Exception as e:
			raise RuntimeError(f"Failed to import ENC data: {str(e)}")

	def get_dataframe(self):
		"""Return the scraped data as a pandas DataFrame."""
		if self.df is None:
			self._scrape_enc_data()
		return self.df

	def save_to_csv(self, filename="ENC_DB.csv"):
		"""Save the DataFrame to a CSV file."""
		if self.df is None:
			self._scrape_enc_data()
		try:
			self.df.to_csv(filename)
		except Exception as e:
			raise IOError(f"Failed to save CSV file: {str(e)}")
