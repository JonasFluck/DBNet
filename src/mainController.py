from MapTypes import MapTypes
from dataDto import DataDto
from dataframeFactory import get_data_frame
from predictionHelper import add_predictions_gauss_regr, add_predictions_knn
from mapCreator import create_map


class MainController:

    @property
    def dto(self):
        return self._dto
    
    @property
    def map_type(self):
        return self._mapType
    
    @property
    def map(self):
        return self._map
    
    def __init__(self, json_data):
        self.json_data = json_data
        self._dto_gauss = None
        self._dto = DataDto(get_data_frame(self.json_data))
        self._mapType = None
    
    def setMap(self,map_type= None, ids= None, state_ids = None):
        print(self._dto.gdf['id'].values)
        self._mapType = map_type
        self._map = create_map(self._dto.gdf, self.map_type, ids, state_ids)
        
