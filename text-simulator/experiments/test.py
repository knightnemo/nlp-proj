from quest_gpt import postProcess
text="""**Final Response**

```json

{

"modified": [

{

"name": "light bulb (ID:2)",

"uuid": 2,

"type": "LightBulb",

"properties": {

"isContainer": false,

"isMoveable": true,

"is_electrical_object": true,

"conductive": true,

"connects": {

"terminal1": ["red wire (ID:3)", "terminal1"],

"terminal2": ["blue wire (ID:5)", "terminal2"]

},

"on": false

},

"contains": []

},

{

"name": "blue wire (ID:5)",

"uuid": 5,

"type": "Wire",

"properties": {

"isContainer": false,

"isMoveable": true,

"is_electrical_object": true,

"conductive": true,

"connects": {

"terminal1": [null, null],

"terminal2": ["light bulb (ID:2)", "terminal2"]

},

"is_wire": true

},

"contains": []

}

],

"removed": [],

"score": {

"score": 0,

"gameOver": false,

"gameWon": false

}

}

```
"""
print(postProcess(text))