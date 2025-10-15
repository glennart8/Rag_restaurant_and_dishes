from pydantic import BaseModel, Field
from typing import List

class PromptStructure(BaseModel):
    name: str | None = None # standardvärdet
    city: str | None = None
    cuisine: str | None = None
    dishes: List[str] = []
    drinks: List[str] = []    

    
class Dish(BaseModel):
    name: str
    ingredients: str
    
class Drink(BaseModel):
    name: str    
    
class Menu(BaseModel):
    dishes: List[Dish]
    drinks: List[Drink]