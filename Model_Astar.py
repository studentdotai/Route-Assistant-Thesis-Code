class Astar:
	def __init__(self, graph):
		self.graph = graph

	@staticmethod
	def heuristic(node1, node2):
		"""
		Euclidean distance heuristic for A* route planning between two nodes.
		Each node is expected to be a tuple (lon, lat).
		"""
		return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)

	def find_nearest_node(self, point):
		"""
		Finds the node in the graph that is closest to the given shapely Point.

		Parameters:
		  point (shapely.geometry.Point): Location from which to search.

		Returns:
		  tuple: A node (as a coordinate tuple) that is nearest to the point.
		"""
		best_node = None
		best_distance = float('inf')
		for node, data in self.graph.nodes(data=True):
			node_point = data.get('point')
			if node_point is None:
				continue
			dist = point.distance(node_point)
			if dist < best_distance:
				best_distance = dist
				best_node = node
		return best_node

	def compute_route(self, start_point, end_point, save_geojson: bool = False, output_file: str = "route.geojson"):
		"""
		Computes the shortest route between start_point and end_point using
		the A* algorithm on the provided graph.

		Parameters:
		  start_point (shapely.geometry.Point): The starting point.
		  end_point (shapely.geometry.Point): The destination point.
		  save_geojson (bool): If True, saves the resulting route in GeoJSON format.
		  output_file (str): File path to save the GeoJSON (default "route.geojson").

		Returns:
		  shapely.geometry.LineString: A LineString representing the computed route.

		Raises:
		  ValueError: If a route cannot be computed.
		"""
		start_node = self.find_nearest_node(start_point)
		print(f"Start Node: {start_node}")
		end_node = self.find_nearest_node(end_point)
		print(f"End Node: {end_node}")
		if start_node is None or end_node is None:
			raise ValueError("Could not find a nearest node for the given start or end point.")
		try:
			route = nx.astar_path(self.graph, start_node, end_node, heuristic=Astar.heuristic, weight='weight')
		except nx.NetworkXNoPath:
			raise ValueError("No route found between the specified start and end points.")

		# Combine Departure and Arrival points with the route
		full_route = [tuple(start_point.coords)[0]] + route + [tuple(end_point.coords)[0]]
		route_linestring = LineString(full_route)
		# Optionally convert the LineString to GeoJSON and save to a file
		if save_geojson:
			geojson_obj = mapping(route_linestring)
			with open(output_file, "w") as f:
				json.dump(geojson_obj, f)
			print(f"Route saved to {output_file}")

		return route_linestring
