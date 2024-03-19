import folium

def generate_map_params(satellite=False):
    """
    Get parameters for folium map
    Parameters
    ----------
    satellite : bool
        True if folium map should have satellite background.
        False if folium map should have openstreetmap background.
    Returns
    -------
    tile: folium map background.
    """
    if satellite == True:
        #network_color = 'white'
        tile = folium.TileLayer(
                    tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                    attr = 'Esri',
                    name = 'Esri Satellite',
                    overlay = False,
                    control = True)
    else:
        tile = 'openstreetmap'   # 'cartodbpositron' is also nice
        #network_color = 'black'
    return tile

def gen_tooltip(fields, aliases):
    """
    Generate tooltip for folium map
    Parameters
    ----------
    fields : list
        column names from dataframe to include in tooltip
    aliases: list
        alias names for fields that show up in tooltip
    Returns
    -------
    tooltip
    """

    tooltip = folium.GeoJsonTooltip(
        fields=fields,
        aliases=aliases,
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border: 2px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """,
        max_width=800,
    )
    return tooltip
