	def create_geo_boundary(self, geometries: Union[gpd.GeoDataFrame, gpd.GeoSeries, List[BaseGeometry]],
							expansion: Union[float, Dict[str, float]] = None,
							crs: int = 4326,
							precision: int = 3,
							date_line:bool = False):
		"""
		 Creates a boundary box (or a MultiPolygon of 2 boxes) from input geometries with optional expansion.

		When `date_line` is True and the original bounds indicate a dateline crossing
		(i.e. the longitudinal span > 180°), this function applies expansion separately for the west and east.

		Args:
			geometries: Input geometries (GeoDataFrame, GeoSeries or list of geometries).
			expansion: Expansion distance in nautical miles. Either a uniform float or a dict with directional keys.
					   For directional expansion, use keys 'W', 'E', 'N', 'S'.
			crs: Coordinate reference system (default: EPSG:4326)
			precision: Decimal precision for rounding boundary coordinates.
			date_line: If True, treat the case where the geometry spans the dateline.

		Returns:
			A GeoDataFrame containing the boundary geometry. In the dateline case a MultiPolygon is returned.
		"""

		def _rnd(_value):
			return round(_value, precision)
		# Convert input to GeoSeries if needed
		if isinstance(geometries, (list, tuple)):
			geom_series = gpd.GeoSeries(geometries)
		elif isinstance(geometries, gpd.GeoDataFrame):
			geom_series = geometries.geometry
		else:
			geom_series = geometries


		# Ensure CRS is set
		geom_series = geom_series.set_crs(crs)

		# Capture the original bounds.
		orig_minx, orig_miny, orig_maxx, orig_maxy = geom_series.total_bounds

		# Define a conversion factor: 1 nautical mile is roughly 1/60 of a degree.
		nm_to_deg = 1 / 60.0

		# Prepare expansion values for directional adjustments.
		if expansion is None:
			exp_w = exp_e = exp_n = exp_s = 0.0
		elif isinstance(expansion, (int, float)):
			exp_w = exp_e = exp_n = exp_s = expansion * nm_to_deg
		elif isinstance(expansion, dict):
			exp_w = expansion.get('W', 0) * nm_to_deg
			exp_e = expansion.get('E', 0) * nm_to_deg
			exp_n = expansion.get('N', 0) * nm_to_deg
			exp_s = expansion.get('S', 0) * nm_to_deg
		else:
			exp_w = exp_e = exp_n = exp_s = 0.0

		# If date_line is requested and the original span suggests dateline crossing.
		if date_line and ((orig_maxx - orig_minx) > 180):
			# Apply expansion separately to each side.
			left_minx = orig_maxx - exp_w
			left_miny = orig_miny - exp_s
			left_maxy = orig_maxy + exp_n
			# Left box extends from left_minx to 180.
			left_box = box(_rnd(left_minx), _rnd(left_miny), 180, _rnd(left_maxy))

			right_maxx = orig_minx + exp_e
			right_miny = orig_miny - exp_s
			right_maxy = orig_maxy + exp_n
			# Right box extends from -180 to right_maxx.
			right_box = box(-180, _rnd(right_miny), _rnd(right_maxx), _rnd(right_maxy))

			multi = MultiPolygon([left_box, right_box])
			bbox_gdf = gpd.GeoDataFrame({
				'geometry': [multi],
				'expansion_type': ['dateline_directional'],
				'expansion_value': [expansion]
			}, crs=crs)
		else:
			# Non‑dateline: Apply expansion uniformly/directionally.
			new_minx = orig_minx - exp_w
			new_miny = orig_miny - exp_s
			new_maxx = orig_maxx + exp_e
			new_maxy = orig_maxy + exp_n
			new_minx = _rnd(new_minx)
			new_miny = _rnd(new_miny)
			new_maxx = _rnd(new_maxx)
			new_maxy = _rnd(new_maxy)
			single_bbox = box(new_minx, new_miny, new_maxx, new_maxy)
			bbox_gdf = gpd.GeoDataFrame({
				'geometry': [single_bbox],
				'expansion_type': ['uniform' if isinstance(expansion, (int, float))
								   else 'directional' if expansion else 'none'],
				'expansion_value': [expansion]
			}, crs=crs)

		return bbox_gdf
