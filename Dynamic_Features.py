class DynamicFeatures():
    def __init__(self, length, breadth, draft, vessel_type, safety_margin, edge_df):
        self.length = length
        self.breadth = breadth
        self.draft = draft
        self.Type= vessel_type
        self.safety_margin = safety_margin
        self.edge_df = edge_df
        self.static_col = None
        self.feature_col = None
        self.dir_col = None
       

    def get_edge_columns(self):

        feature = [col[3:] for col in self.edge_df.columns if col.startswith('ft_')]
        self.feature_col=['wt_'+ col for col in feature]
        self.dir_col = [col for col in self.edge_df.columns if 'directional' in col]
        self.static_col = [col for col in self.edge_df.columns if col.startswith('wt_') and 
                           col not in self.feature_col and col not in self.dir_col]  
    
        print(f"Static Columns: {self.static_col}")
        print(f"Feature Columns: {self.feature_col}")
        print(f"Directional Columns: {self.dir_col}")
    
    @staticmethod
    def export_edges_to_geopackage(edges_gdf, filepath, layer_name='edges', driver='GPKG'):
        """
        Export an edge table (GeoDataFrame) to a GeoPackage file with a custom name.
        
        Parameters:
        -----------
        edges_gdf : geopandas.GeoDataFrame
            The GeoDataFrame containing edge geometries and attributes to export
        filepath : str
            The full path to the output file (including .gpkg extension)
        layer_name : str, optional
            The name of the layer within the GeoPackage (default: 'edges')
        driver : str, optional
            The OGR driver to use for the export (default: 'GPKG' for GeoPackage)
            
        Returns:
        --------
        bool
            True if the export was successful, False otherwise
        """
        try:
            # Make sure we have a GeoDataFrame with valid geometries
            if not isinstance(edges_gdf, gpd.GeoDataFrame):
                raise TypeError("Input must be a GeoDataFrame")
            
            # Create a copy to avoid modifying the original
            edges_to_export = edges_gdf.copy()
            
            # Check for null geometries
            if edges_to_export.geometry.isna().any():
                print(f"Warning: {edges_to_export.geometry.isna().sum()} rows have null geometries")
                # Optionally remove rows with null geometries
                edges_to_export = edges_to_export.dropna(subset=['geometry'])
            
            # Check for invalid geometries and try to fix them
            invalid_geoms = ~edges_to_export.geometry.is_valid
            if invalid_geoms.any():
                print(f"Warning: {invalid_geoms.sum()} geometries are invalid. Attempting to fix...")
                edges_to_export.geometry = edges_to_export.geometry.buffer(0)  # Common fix for invalid geometries
            
            # Ensure CRS is defined
            if edges_to_export.crs is None:
                print("Warning: CRS is not defined. Setting to EPSG:4326 (WGS84).")
                edges_to_export.crs = "EPSG:4326"
            
            # Convert complex data types to strings to ensure compatibility
            for col in edges_to_export.columns:
                if edges_to_export[col].dtype.name == 'object':
                    # Convert any non-string objects to strings
                    edges_to_export[col] = edges_to_export[col].astype(str)
            
            # Ensure the filepath has the correct extension
            if not filepath.lower().endswith('.gpkg'):
                filepath = f"{filepath}.gpkg"
            
            # Export to GeoPackage
            edges_to_export.to_file(filepath, layer=layer_name, driver=driver)
            
            print(f"Successfully exported {len(edges_to_export)} edges to {filepath} (layer: {layer_name})")
            return True
        
        except Exception as e:
            print(f"Error exporting edges to GeoPackage: {str(e)}")
            return False

    def calculate_depth_weight(self, draft=None, safety_margin=None):
        """
        Calculate wt_depth_min based on the difference between ft_depth_min and vessel draft.
        
        Parameters:
        -----------
        edges_df : pandas.DataFrame
            DataFrame containing edge information with ft_depth_min column
        draft : float
            Vessel draft in the same units as ft_depth_min
        safety_margin : float, optional
            Additional safety margin to add (default: 0.0)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added wt_depth_min column
        """
        if draft is None:
            draft = self.draft
        if safety_margin is None:
            safety_margin = self.safety_margin


        # Create a copy to avoid modifying the original DataFrame
        df = self.edge_df.copy()
        
        # Calculate water depth (ft_depth_min - draft)
        df["calc_depth"] = df["ft_drgare"].fillna(df["ft_depth_min"])
        ukc = df["calc_depth"] - draft
    
        # Optional: Apply some weighting logic
        df.loc[ukc <= 0, "wt_depth_min"] = float(999)  # Infinite weight for unsafe passages
        
        # For shallow but passable water, you might want to apply a non-linear weight
        shallow = (ukc > 0) & (ukc < safety_margin)  # Example threshold
        df.loc[shallow, "wt_depth_min"] = 1.1
        safe = (ukc > safety_margin) & (ukc <= 15)  # Example threshold
        df.loc[safe, "wt_depth_min"] = 1.0
        # For deep water, minimal weight
        df.loc[ukc > 15, "wt_depth_min"] = 0.9
        self.edge_df = df
        return df

    def update_dynamic_factor(self, weight_columns=None):
        """
        Update the dynamic factor column by multiplying weights from specified columns.
        
        Parameters:
        -----------
        edges_df : pandas.DataFrame
            DataFrame containing edge information with weight columns
        weight_columns : list, optional
            List of weight column names to include in the calculation.
            If None, uses default list of weight columns.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with updated dynamic factor column
        """
        # Create a copy to avoid modifying the original DataFrame
        df = self.edge_df.copy()
        
        # Default weight columns if none provided
        if weight_columns is None:
            self.get_edge_columns()
            weight_columns = self.feature_col
        
        # Initialize dynamic factor column with 1.0
        df['dynamic_factor'] = 1.0
        
        # Multiply all specified weight columns
        for col in weight_columns:
            if col in df.columns:
                df['dynamic_factor'] *= df[col]
            else:
                print(f"Warning: Column '{col}' not found in DataFrame")
        self.edge_df = df
        return df
    
    def calculate_adjusted_weights(self, base_weight_col='base_weight', 
                              static_factor_col='static_weight_factor', 
                              dynamic_factor_col='dynamic_factor',
                              output_col='adjusted_weight'):
        """
        Calculate adjusted weights by combining base weight, static weight factor, and dynamic factor.
        
        Parameters:
        -----------
        edges_df : pandas.DataFrame
            DataFrame containing edge information with weight columns
        base_weight_col : str, optional
            Column name for base weight (default: 'base_weight')
        static_factor_col : str, optional
            Column name for static weight factor (default: 'static_weight_factor')
        dynamic_factor_col : str, optional
            Column name for dynamic factor (default: 'dynamic_factor')
        output_col : str, optional
            Column name for the calculated adjusted weight (default: 'adjusted_weight')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added adjusted weight column
        """
        # Create a copy to avoid modifying the original DataFrame
        df = self.edge_df.copy()
        
        # Check if required columns exist
        required_cols = [base_weight_col, static_factor_col, dynamic_factor_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate adjusted weight
        # Formula: adjusted_weight = base_weight * static_weight_factor * dynamic_factor
        df[output_col] = df[base_weight_col] * df[static_factor_col] * df[dynamic_factor_col]
        self.edge_df = df
        return df
    
    def compare_adjusted_weights(self, original_df, updated_df=None, weight_col='adjusted_weight', 
                            id_cols=['source', 'target'], tolerance=1e-10):
            """
            Compare adjusted weights between original and updated edge dataframes.
            
            Parameters:
            -----------
            original_df : pandas.DataFrame
                Original DataFrame containing edge information with weight column
            updated_df : pandas.DataFrame
                Updated DataFrame containing edge information with weight column
            weight_col : str, optional
                Column name for the adjusted weight (default: 'adjusted_weight')
            id_cols : list, optional
                Columns that uniquely identify an edge (default: ['source', 'target'])
            tolerance : float, optional
                Numerical tolerance for comparing float values (default: 1e-10)
                
            Returns:
            --------
            dict
                Dictionary containing statistics about the weight changes:
                - 'total_edges': Total number of edges
                - 'changed_edges': Number of edges with changed weights
                - 'unchanged_edges': Number of edges with unchanged weights
                - 'percent_changed': Percentage of edges that changed
                - 'changes_df': DataFrame showing original and new weights for changed edges
            """
            if updated_df is None:
                updated_df = self.edge_df
            # Check if required columns exist in both dataframes
            required_cols = id_cols + [weight_col]
            for df, name in [(original_df, 'original_df'), (updated_df, 'updated_df')]:
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns in {name}: {missing_cols}")
            
            # Ensure both dataframes have the same edges
            if len(original_df) != len(updated_df):
                raise ValueError(f"Dataframes have different number of edges: original={len(original_df)}, updated={len(updated_df)}")
            
            # Merge dataframes on id columns to compare weights
            merged_df = original_df.merge(
                updated_df[id_cols + [weight_col]], 
                on=id_cols, 
                suffixes=('_original', '_updated')
            )
            
            # Calculate absolute difference
            original_col = f"{weight_col}_original"
            updated_col = f"{weight_col}_updated"
            merged_df['weight_diff'] = abs(merged_df[original_col] - merged_df[updated_col])
            
            # Identify changed edges (difference greater than tolerance)
            changed_edges = merged_df[merged_df['weight_diff'] > tolerance]
            
            # Calculate statistics
            total_edges = len(merged_df)
            num_changed = len(changed_edges)
            num_unchanged = total_edges - num_changed
            percent_changed = (num_changed / total_edges) * 100 if total_edges > 0 else 0
            
            # Create a dataframe with changes for inspection
            changes_df = changed_edges[id_cols + [original_col, updated_col, 'weight_diff']]
            changes_df = changes_df.sort_values('weight_diff', ascending=False)
            

            print(f"total_edges: {total_edges}")
            print(f"num_changed: {num_changed}")
            print(f"num_unchanged: {num_unchanged}")
            print(f"percent_changed: {percent_changed}")
            # Return statistics
            return  changes_df
