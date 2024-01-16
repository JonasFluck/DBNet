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
    
    def __init__(self):
        self._dto = None
        self._mapType = None
    
    def setData(self,json_data, mapType, ids = None):
        self._mapType = mapType
        
        if(mapType == MapTypes.Gauss):
            self._dto = DataDto(add_predictions_gauss_regr(get_data_frame(json_data)))
        elif(mapType == MapTypes.KNN):
            self._dto = DataDto(add_predictions_knn(get_data_frame(json_data, ids)))
        else:
            self._dto = DataDto(get_data_frame(json_data, ids)) 
        self._map = create_map(self._dto.gdf, mapType)
