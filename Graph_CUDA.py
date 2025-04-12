def create_graph_V2(nodes_df, edges_df,  source_col='source', target_col='target', weight_col='weight', directed=False, use_cuda=True, backend="cugraph"):
    """
    Creates a graph containing only the largest connected component directly from an edge list.
    
    This function first creates a temporary graph to identify connected components,
    then filters the edge list to only include edges from the largest component,
    and finally creates the final graph from the filtered edge list.
    
    Parameters:
        edges_df: DataFrame or cuDF DataFrame containing edge information
        nodes_df: DataFrame or cuDF DataFrame containing node information (optional)
        source_col: Column name for source nodes
        target_col: Column name for target nodes
        directed: Whether to create a directed graph
        use_cuda: Whether to use CUDA acceleration with cuGraph
        
    Returns:
        Tuple of (graph, filtered_nodes_df, filtered_edges_df) where:
            - graph: A graph (cuGraph or NetworkX) containing only the largest connected component
            - filtered_nodes_df: DataFrame containing only nodes in the largest component
            - filtered_edges_df: DataFrame containing only edges in the largest component
    """
    # Determine if we're using CUDA
    if use_cuda:
        try:
            import cudf
            import cugraph
            
            # Convert to cuDF if it's a pandas DataFrame
            if not isinstance(edges_df, cudf.DataFrame):
                edges_df = edges_df.copy()
                # Remove geometry column if it exists
                if 'geometry' in edges_df.columns:
                    edges_df.drop(columns=["geometry"], inplace=True)
                edges_df = cudf.DataFrame.from_pandas(edges_df)
                print(f"Created CuDF DataFrame from pandas Edges DataFrame.\nLength: {len(edges_df)}")
            
            # Convert nodes_df to cuDF if provided and it's a pandas DataFrame
            if nodes_df is not None and not isinstance(nodes_df, cudf.DataFrame):
                nodes_df = nodes_df.copy()
                # Remove geometry column if it exists
                if 'geometry' in nodes_df.columns:
                    nodes_df.drop(columns=["geometry"], inplace=True)
                nodes_df = cudf.DataFrame.from_pandas(nodes_df)
                print(f"Created CuDF DataFrame from pandas Nodes DataFrame.\nLength: {len(nodes_df)}")
            
            # Create a temporary graph to find connected components
            temp_graph = cugraph.Graph(directed=False)
            temp_graph.from_cudf_edgelist(edges_df, source=source_col, destination=target_col)
            
            # Find connected components
            components = cugraph.connected_components(temp_graph)
            
            # Count sizes of each component
            component_counts = components.groupby('labels').size().reset_index()
            component_counts.columns = ['labels', 'size']
            
            # Sort by size in descending order
            component_counts = component_counts.sort_values('size', ascending=False)
            
            # The largest component label is now at index 0
            largest_component_label = component_counts['labels'].iloc[0]
            print(f"Largest Component Label: {largest_component_label}")
            print(f"Largest Component Size: {component_counts['size'].iloc[0]}")
            
            # Filter nodes to keep only those in the largest component
            nodes_to_keep = components[components['labels'] == largest_component_label]['vertex']
            
            # Filter edges to keep only those where both endpoints are in the largest component
            filtered_edges = edges_df[
                edges_df[source_col].isin(nodes_to_keep) & 
                edges_df[target_col].isin(nodes_to_keep)
            ]
            
            # Filter nodes_df if provided
            filtered_nodes = None
            if nodes_df is not None:
                # Assuming the node ID column is named 'id' - adjust if different
                node_id_col = 'node' if 'node' in nodes_df.columns else 'id'
                print(f"Node id column found: {node_id_col}")
                filtered_nodes = nodes_df[nodes_df[node_id_col].isin(nodes_to_keep)]
                print(f"Nodes processed: {len(filtered_nodes)}")
            
            # Create the final graph from the filtered edge list
            final_graph = cugraph.Graph(directed=directed)
            essential_attrs = ["forward_heading", "reverse_heading"]  # Add only the attributes you need
            edge_attrs = [col for col in filtered_edges.columns if col in essential_attrs and col not in [source_col, target_col]]
            final_graph.from_cudf_edgelist(
                filtered_edges, 
                source=source_col, 
                destination=target_col,
                weight = weight_col,
                symmetrize=True,
                
            )
            #edge_attr=edge_attrs if edge_attrs else None
            
            return final_graph, filtered_nodes, filtered_edges
            
        except (ImportError, Exception) as e:
            print(f"CUDA graph creation failed: {e}")
            use_cuda = False
    
    # Fallback to NetworkX
    if not use_cuda:
        import networkx as nx
        import nx_cugraph as nxcg
        import pandas as pd
        
        # Convert to pandas if it's a cuDF DataFrame
        if not isinstance(edges_df, pd.DataFrame):
            edges_df = edges_df.to_pandas()
        
        # Convert nodes_df to pandas if provided and it's a cuDF DataFrame
        if nodes_df is not None and not isinstance(nodes_df, pd.DataFrame):
            nodes_df = nodes_df.to_pandas()
        
        # Create a temporary graph
        if directed:
            temp_graph = nx.DiGraph()
        else:
            temp_graph = nx.Graph()
            

            
        # Add edges from the DataFrame
        for _, row in edges_df.iterrows():
            temp_graph.add_edge(row[source_col], row[target_col])
        
        # Find the largest connected component
        if directed:
            largest_cc = max(nx.weakly_connected_components(temp_graph), key=len)
        else:
            largest_cc = max(nx.connected_components(temp_graph), key=len)
        
        # Filter edges to keep only those in the largest component
        filtered_edges = edges_df[
            edges_df[source_col].isin(largest_cc) & 
            edges_df[target_col].isin(largest_cc)
        ]
        
        # Filter nodes_df if provided
        filtered_nodes = None
        if nodes_df is not None:
            # Assuming the node ID column is named 'id' - adjust if different
            node_id_col = 'node' if 'node' in nodes_df.columns else 'id'
            print(f"Node ID column: {node_id_col}")
            filtered_nodes = nodes_df[nodes_df[node_id_col].isin(largest_cc)]
            print(f"Filtered nodes: {len(filtered_nodes)}")
        
        # Create the final graph
        if directed:
            final_graph = nx.DiGraph()
        else:
            final_graph= nx.Graph()
            
            
            
        # Add edges and attributes from the filtered DataFrame
        for _, row in filtered_edges.iterrows():
            attrs = {col: row[col] for col in filtered_edges.columns 
                    if col not in [source_col, target_col]}
            final_graph.add_edge(row[source_col], row[target_col], **attrs)
            
        return final_graph, filtered_nodes, filtered_edges
