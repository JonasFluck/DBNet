from providerHelper import avg_for_provider


class DataDto:
    def __init__(self, gdf):
        self._gdf = gdf
        self._avg_providers = avg_for_provider(gdf)
        
    @property
    def gdf(self):
        return self._gdf
    
    @property
    def avg_providers(self):
        return self._avg_providers