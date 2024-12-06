# use-bandage.py
# based on make-campfire.py
# peter jansen (feb 7/2023)

# Task: Create a micro-simulation that requires a user to put bandage on any cuts.
# Environment: bathroom
# Task-critical Objects: BandageBox, Bondage, Person, BodyPart
# High-level object classes: Container (BondageBox, Person, BodyPart)
# Critical properties: hasWound (BodyPart)
# Actions: look, inventory, examine, take/put object, open/close container
# Distractor Items: Sticker, Clothing
# Distractor Actions: None
# High-level solution procedure: open the bandage box, take the bandage, put the bandage on the wounded part of the person

import random

UUID = 0
randomSeed = 1
#
# Abstract class for all game objects
#
class GameObject():
    def __init__(self, name):
        # Prevent this constructor from running if it's already been run during multiple inheritance
        if hasattr(self, "constructorsRun"):
            return
        # Otherwise, keep a list of constructors that have already been run
        self.constructorsRun = ["GameObject"]
        global UUID
        self.uuid = UUID
        UUID += 1

        self.name = f"{name} (ID: {self.uuid})"
        self.parentContainer = None
        self.contains = []
        self.properties = {}

        # Set critical properties
        self.properties["isContainer"] = False    # By default, objects are not containers
        self.properties["isMoveable"] = True     # By default, objects are moveable

    # Get a property of the object (safely), returning None if the property doesn't exist
    def getProperty(self, propertyName):
        if propertyName in self.properties:
            return self.properties[propertyName]
        else:
            return None

    # Add an object to this container, removing it from its previous container
    def addObject(self, obj):
        obj.removeSelfFromContainer()
        self.contains.append(obj)
        obj.parentContainer = self

    # Remove an object from this container
    def removeObject(self, obj):
        self.contains.remove(obj)
        obj.parentContainer = None

    # Remove the current object from whatever container it's currently in
    def removeSelfFromContainer(self):
        if self.parentContainer != None:
            self.parentContainer.removeObject(self)

    # Get all contained objects, recursively
    def getAllContainedObjectsRecursive(self):
        outList = []
        for obj in self.contains:
            # Add self
            outList.append(obj)
            # Add all contained objects
            outList.extend(obj.getAllContainedObjectsRecursive())
        return outList

    # Get all contained objects that have a specific name (not recursively)
    def containsItemWithName(self, name):
        foundObjects = []
        for obj in self.contains:
            if obj.name == name:
                foundObjects.append(obj)
        return foundObjects

    # Game tick: Perform any internal updates that need to be performed at each step of the game.
    def tick(self):
        # If this object is combusting, then reduce the time remaining
        if self.getProperty("isCombusting") == True:
            self.properties["combustionTimeRemaining"] -= 1
            if self.properties["combustionTimeRemaining"] <= -1:        # -1 signifies that the object has combusted.
                self.properties["isCombusting"] = False


        pass

    # Get a list of referents (i.e. names that this object can be called by)
    def getReferents(self):
        return [self.name]

    def getCanonicalReferent(self):
        return self.getReferents()[0]

    # Make a human-readable string that describes this object
    def makeDescriptionStr(self, makeDetailed=False):
        return self.name


#
#   Abstract Game-object Classes
#


# Abstract class for things that can be considered 'containers' (e.g. a drawer, a box, a table, a shelf, etc.)
class Container(GameObject):
    def __init__(self, name):
        # Prevent this constructor from running if it's already been run during multiple inheritance
        if hasattr(self, "constructorsRun"):
            if "Container" in self.constructorsRun:
                return

        GameObject.__init__(self, name)
        # Otherwise, mark this constructor as having been run
        self.constructorsRun.append("Container")

        # Set critical properties
        self.properties["isContainer"] = True
        self.properties["isOpenable"] = False  # Can the container be opened (e.g. a drawer, a door, a box, etc.), or is it always 'open' (e.g. a table, a shelf, etc.)
        self.properties["isOpen"] = True      # Is the container open or closed (if it is openable)
        self.properties["containerPrefix"] = "in" # The prefix to use when referring to the container (e.g. "in the drawer", "on the table", etc.)

    # Try to open the container
    # Returns an observation string, and a success flag (boolean)
    def openContainer(self):
        # First, check to see if this object is openable
        if not self.getProperty("isOpenable"):
            # If not, then it can't be opened
            return ("The " + self.name + " can't be opened.", False)

        # If this object is openable, then check to see if it is already open
        if self.getProperty("isOpen"):
            # If so, then it can't be opened
            return ("The " + self.name + " is already open.", False)

        # If this object is openable and it is closed, then open it
        self.properties["isOpen"] = True
        return ("The " + self.name + " is now open.", True)

    # Try to close the container
    # Returns an observation string, and a success flag (boolean)
    def closeContainer(self):
        # First, check to see if this object is openable
        if not (self.getProperty("isOpenable") == True):
            # If not, then it can't be closed
            return ("The " + self.name + " can't be closed.", False)

        # If this object is openable, then check to see if it is already closed
        if not (self.getProperty("isOpen") == True):
            # If so, then it can't be closed
            return ("The " + self.name + " is already closed.", False)

        # If this object is openable and it is open, then close it
        self.properties["isOpen"] = False
        return ("The " + self.name + " is now closed.", True)

    # Try to place the object in a container.
    # Returns an observation string, and a success flag (boolean)
    def placeObjectInContainer(self, obj):
        # First, check to see if this object is a container
        if not (self.getProperty("isContainer") == True):
            # If not, then it can't be placed in a container
            return ("The " + self.name + " is not a container, so things can't be placed there.", False)

        # Check to see if the object is moveable
        if not (obj.getProperty("isMoveable") == True):
            # If not, then it can't be removed from a container
            return ("The " + obj.name + " is not moveable.", None, False)

        # If this object is a container, then check to see if it is open
        if not (self.getProperty("isOpen") == True):
            # If not, then it can't be placed in a container
            return ("The " + self.name + " is closed, so things can't be placed there.", False)

        # If this object is a container and it is open, then place the object in the container
        self.addObject(obj)
        return ("The " + obj.getReferents()[0] + " is placed in the " + self.name + ".", True)

    # Try to remove the object from a container.
    # Returns an observation string, a reference to the object being taken, and a success flag (boolean)
    def takeObjectFromContainer(self, obj):
        # First, check to see if this object is a container
        if not (self.getProperty("isContainer") == True):
            # If not, then it can't be removed from a container
            return ("The " + self.name + " is not a container, so things can't be removed from it.", None, False)

        # Check to see if the object is moveable
        if not (obj.getProperty("isMoveable") == True):
            # If not, then it can't be removed from a container
            return ("The " + obj.name + " is not moveable.", None, False)

        # If this object is a container, then check to see if it is open
        if not (self.getProperty("isOpen") == True):
            # If not, then it can't be removed from a container
            return ("The " + self.name + " is closed, so things can't be removed from it.", None, False)

        # Check to make sure that the object is contained in this container
        if obj not in self.contains:
            return ("The " + obj.name + " is not contained in the " + self.name + ".", None, False)

        # If this object is a container and it is open, then remove the object from the container
        obj.removeSelfFromContainer()
        return ("The " + obj.getReferents()[0] + " is removed from the " + self.name + ".", obj, True)

    # Make a human-readable string that describes this object
    def makeDescriptionStr(self, makeDetailed=False):
        return "the " + self.name + "."


#
#   Specific Game Objects
#

# A box of bandages
class BandageBox(Container):
    # Constructor.
    def __init__(self):
        GameObject.__init__(self, "bandage box")
        Container.__init__(self, "bandage box")

        self.properties["containerPrefix"] = "in"
        # Set the properties of this object
        self.properties["isOpenable"] = True    # A bandage box can be opened and closed
        self.properties["isOpen"] = False       # A bandage box is closed by default

    def makeDescriptionStr(self, makeDetailed=False):
        outStr = f"a {self.name}"

        # If the box is closed, then say it's closed.  Otherwise, say what's inside it.
        if (self.getProperty("isOpen") == False):
            outStr += " that is closed"
        else:
            outStr += " that is open"

            effectiveContents = []
            for obj in self.contains:
                effectiveContents.append(obj.makeDescriptionStr())

            if (len(effectiveContents) > 0):
                outStr += " and that looks to have "
                for i in range(len(effectiveContents)):
                    if (i == len(effectiveContents) - 1) and (len(effectiveContents) > 1):
                        outStr += "and "
                    outStr += effectiveContents[i] + ", "
                outStr = outStr[:-2] + " " + self.properties["containerPrefix"] + " it"
            else:
                outStr += " and appears to be empty"

        return outStr

# A bandage
class Bandage(GameObject):
    # Constructor
    def __init__(self):
        GameObject.__init__(self, "bandage")

    # Make a human-readable string that describes this object
    def makeDescriptionStr(self, makeDetailed=False):
        return f"a {self.name}"


# A person, that contains different body parts
class Person(Container):
    # Constructor
    def __init__(self):
        Container.__init__(self, "person")
        self.properties["containerPrefix"] = "on"       # Bandages go "on" a body

    # Make a human-readable string that describes this object
    def makeDescriptionStr(self, makeDetailed=False):
        outStr = "a " + self.name

        # List contents
        effectiveContents = []
        for obj in self.contains:
            effectiveContents.append(obj.makeDescriptionStr())

        if (len(effectiveContents) > 0):
            outStr += " that looks to have "
            for i in range(len(effectiveContents)):
                if (i == len(effectiveContents) - 1) and (len(effectiveContents) > 1):
                    outStr += "and "
                outStr += effectiveContents[i] + ", "
            outStr = outStr[:-2] # No container prefix for people (sounds unusual)

        return outStr


# A body part
class BodyPart(Container):
    # Constructor
    def __init__(self, bodyPartName, hasWound=False):
        Container.__init__(self, bodyPartName)
        self.properties["containerPrefix"] = "on"       # Bandages go "on" a body part
        # Set critical properties
        self.properties["hasWound"] = hasWound          # Does this body part have a wound that needs a bandage?
        self.properties["isMoveable"] = False

    # Special override for putting objects on body parts, that checks that clothing is being put on the right body part
    # Try to place the object in a container.
    # Returns an observation string, and a success flag (boolean)
    def placeObjectInContainer(self, obj):

        # Check to see if the object is an instance of Clothing
        if isinstance(obj, Clothing):
            # If so, check to see if the clothing is appropriate for this body part
            if not (self.name in obj.properties["bodyPartItFits"]):
                return ("The " + obj.name + " is not appropriate for the " + self.name + ".", False)

        # Otherwise, call the base class method
        return Container.placeObjectInContainer(self, obj)

    # Make a human-readable string that describes this object
    def makeDescriptionStr(self, makeDetailed=False):
        outStr = "a " + self.name

        # If the body part has a wound, then say it has a wound.  Otherwise, say what's on it.
        if (self.getProperty("hasWound") == True):
            outStr += " that has a wound"

        # If the body part has a bandage on it, then say it has a bandage on it.
        foundBandages = self.containsItemWithName("bandage")
        if (len(foundBandages) == 1):
            outStr += " with a bandage on it"
        elif (len(foundBandages) > 1):
            outStr += " with bandages on it"

        # If the body part has something else on it, then say what it is.
        effectiveContents = []
        for obj in self.contains:
            if (obj.name != "bandage"):
                effectiveContents.append(obj.makeDescriptionStr())

        if (len(effectiveContents) > 0):
            outStr += " that has "
            for i in range(len(effectiveContents)):
                if (i == len(effectiveContents) - 1) and (len(effectiveContents) > 1):
                    outStr += "and "
                outStr += effectiveContents[i] + ", "
            outStr = outStr[:-2] + " " + self.properties["containerPrefix"] + " it"

        return outStr


# Distractor items

# A sticker, that has a description of what it looks like
class Sticker(GameObject):
    # Constructor
    def __init__(self, stickerDescription):
        GameObject.__init__(self, "sticker")
        # Set critical properties
        self.properties["stickerDescription"] = stickerDescription

    # Make a human-readable string that describes this object
    def makeDescriptionStr(self, makeDetailed=False):
        return f"a {self.name}{self.properties['stickerDescription']}"

# A piece of clothing
class Clothing(GameObject):
    # Constructor
    def __init__(self, name, bodyPartItFits:list):
        GameObject.__init__(self, name)
        # Set critical properties
        self.properties["bodyPartItFits"] = bodyPartItFits

    # Make a human-readable string that describes this object
    def makeDescriptionStr(self, makeDetailed=False):
        return "a " + self.name



# The world is the root object of the game object tree.  In single room environments, it's where all the objects are located.
class World(Container):
    def __init__(self):
        Container.__init__(self, "bathroom")

    def makeDescriptionStr(self, makeDetailed=False):
        outStr = "You find yourself in a bathroom.  Around you, you see: \n"
        for obj in self.contains:
            outStr += "\t" + obj.makeDescriptionStr() + "\n"

        return outStr


# The agent (just a placeholder for a container for the inventory)
class Agent(Container):
    def __init__(self):
        GameObject.__init__(self, "agent")
        Container.__init__(self, "agent")

    def getReferents(self):
        return ["yourself"]

    def makeDescriptionStr(self, makeDetailed=False):
        return "yourself"


class TextGame:

    def __init__(self, randomSeed):
        # Random number generator, initialized with a seed passed as an argument
        self.random = random.Random(randomSeed)
        # Reset global UUID
        global UUID
        UUID = 0
        # The agent/player
        self.agent = Agent()
        # Game Object Tree
        self.rootObject = self.initializeWorld()
        # Game score
        self.score = 0
        self.numSteps = 0
        # Game over flag
        self.gameOver = False
        self.gameWon = False
        # Last game observation
        self.observationStr = self.rootObject.makeDescriptionStr()
        # Do calculate initial scoring
        self.calculateScore()

    # Create/initialize the world/environment for this game
    def initializeWorld(self):
        world = World()

        # Add the agent
        world.addObject(self.agent)

        # Add a person
        person = Person()
        world.addObject(person)

        # Add body parts
        bodyPartNames = ["left hand", "right hand", "left foot", "right foot", "head"]
        # Randomly shuffle the body parts (the first will have a wound)
        self.random.shuffle(bodyPartNames)
        for idx, bodyPartName in enumerate(bodyPartNames):
            hasWound = False
            if (idx == 0):
                hasWound = True
            bodyPart = BodyPart(bodyPartName, hasWound=hasWound)
            person.addObject(bodyPart)
        # Randomize the presentation of the body parts, so it's not obvious from the order which one has the wound
        self.random.shuffle(person.contains)


        # Add bandage box
        bandageBox = BandageBox()
        world.addObject(bandageBox)

        # Add a bandage
        bandage = Bandage()
        bandageBox.addObject(bandage)

        # TODO: Distractor items
        stickerMessages = [" with a picture of a cat", " with a picture of a dog", " with a picture of a fish", " that is pink", " that says 'get well soon'", " that says 'I donated blood'"]
        self.random.shuffle(stickerMessages)
        # Add a sticker
        sticker = Sticker(stickerMessages[0])
        world.addObject(sticker)

        # Add a piece of clothing
        distractorClothing = [
            Clothing("blue glove", ["left hand", "right hand"]),
            Clothing("red glove", ["left hand", "right hand"]),
            Clothing("warm stock", ["left foot", "right foot"]),
            Clothing("pink hat", ["head"]),
            Clothing("baseball hat", ["head"]),
        ]
        self.random.shuffle(distractorClothing)
        world.addObject(distractorClothing[0])

        # Shuffle the objects in the world
        self.random.shuffle(world.contains)

        # Return the world
        return world

    # Get the task description for this game
    def getTaskDescription(self):
        return "Your task is to put bandages on any cuts."

    # Make a dictionary whose keys are object names (strings), and whose values are lists of object references with those names.
    # This is useful for generating valid actions, and parsing user input.
    def makeNameToObjectDict(self):
        # Get a list of all game objects
        allObjects = self.rootObject.getAllContainedObjectsRecursive()

        # Make a dictionary whose keys are object names (strings), and whose values are lists of object references with those names.
        nameToObjectDict = {}
        for obj in allObjects:
            for name in obj.getReferents():
                #print("Object referent: " + name)
                if name in nameToObjectDict:
                    nameToObjectDict[name].append(obj)
                else:
                    nameToObjectDict[name] = [obj]

        return nameToObjectDict

    #
    #   Action generation
    #

    def addAction(self, actionStr, actionArgs):
        # Check whether the action string key already exists -- if not, add a blank list
        if not (actionStr in self.possibleActions):
            self.possibleActions[actionStr] = []
        # Add the action arguments to the list
        self.possibleActions[actionStr].append(actionArgs)

    # Returns a list of valid actions at the current time step
    def generatePossibleActions(self):
        # Get a list of all game objects that could serve as arguments to actions
        allObjects = self.makeNameToObjectDict()

        # Make a dictionary whose keys are possible action strings, and whose values are lists that contain the arguments.
        self.possibleActions = {}

        # Actions with zero arguments
        # (0-arg) Look around the environment
        self.addAction("look around", ["look around"])
        self.addAction("look", ["look around"])

        # (0-arg) Look at the agent's current inventory
        self.addAction("inventory", ["inventory"])

        # Actions with one object argument
        # (1-arg) Take
        for objReferent, objs in allObjects.items():
            for obj in objs:
                self.addAction("take " + objReferent, ["take", obj])
                self.addAction("take " + objReferent + " from " + obj.parentContainer.getReferents()[0], ["take", obj])

        # (1-arg) Open/Close
        ## (OMIT, UNUSED HERE, make-campfire)
        for objReferent, objs in allObjects.items():
            for obj in objs:
                self.addAction("open " + objReferent, ["open", obj])
                self.addAction("close " + objReferent, ["close", obj])

        # (1-arg) Detailed look/examine
        for objReferent, objs in allObjects.items():
            for obj in objs:
                self.addAction("examine " + objReferent, ["examine", obj])

        # Actions with two object arguments
        # (2-arg) Put
        for objReferent1, objs1 in allObjects.items():
            for objReferent2, objs2 in allObjects.items():
                for obj1 in objs1:
                    for obj2 in objs2:
                        if (obj1 != obj2):
                            containerPrefix = "in"
                            if obj2.properties["isContainer"]:
                                containerPrefix = obj2.properties["containerPrefix"]
                            self.addAction("put " + objReferent1 + " " + containerPrefix + " " + objReferent2, ["put", obj1, obj2])

        return self.possibleActions

    #
    #   Interpret actions
    #

    # Describe the room that the agent is currently in
    def actionLook(self):
        return self.agent.parentContainer.makeDescriptionStr()

    # Open a container
    def actionOpen(self, obj):
        # Check if the object is a container
        if (obj.getProperty("isContainer") == True):
            # This is handled by the object itself
            obsStr, success = obj.openContainer()
            return obsStr
        else:
            return "You can't open that."

    # Close a container
    def actionClose(self, obj):
        # Check if the object is a container
        if (obj.getProperty("isContainer") == True):
            # This is handled by the object itself
            obsStr, success = obj.closeContainer()
            return obsStr
        else:
            return "You can't close that."

    # Take an object from a container
    def actionTake(self, obj):
        # If the object doesn't have a parent container, then it's dangling and something has gone wrong
        if (obj.parentContainer == None):
            return "Something has gone wrong -- that object is dangling in the void.  You can't take that."

        # Cannot take the agent
        if type(obj) == Agent:
            return "You cannot take yourself."
        
        # You can't pick up objects that are combusting
        if (obj.getProperty("isCombusting") == True):
            return "You can't pick up the " + obj.getReferents()[0] + " because it's on fire!"

        # Take the object from the parent container, and put it in the inventory
        obsStr, objRef, success = obj.parentContainer.takeObjectFromContainer(obj)
        if (success == False):
            return obsStr

        # Add the object to the inventory
        self.agent.addObject(obj)
        return obsStr + " You put the " + obj.getReferents()[0] + " in your inventory."

    # Put an object in a container
    def actionPut(self, objToMove, newContainer):
        # Check that the destination container is a container
        if (newContainer.getProperty("isContainer") == False):
            return "You can't put things in the " + newContainer.getReferents()[0] + "."

        # Enforce that the object must be in the inventory to do anything with it
        if (objToMove.parentContainer != self.agent):
            return "You don't currently have the " + objToMove.getReferents()[0] + " in your inventory."

        # Take the object from it's current container, and put it in the new container.
        # Deep copy the reference to the original parent container, because the object's parent container will be changed when it's taken from the original container
        originalContainer = objToMove.parentContainer
        obsStr1, objRef, success = objToMove.parentContainer.takeObjectFromContainer(objToMove)
        if (success == False):
            return obsStr1

        # Put the object in the new container
        obsStr2, success = newContainer.placeObjectInContainer(objToMove)
        if (success == False):
            # For whatever reason, the object can't be moved into the new container. Put the object back into the original container
            originalContainer.addObject(objToMove)
            return obsStr2

        # Success -- show both take and put observations
        return obsStr1 + "\n" + obsStr2


    # Display agent inventory
    def actionInventory(self):
        # Get the inventory
        inventory = self.agent.contains
        # If the inventory is empty, return a message
        if (len(inventory) == 0):
            return "Your inventory is empty."
        # Otherwise, return a list of the inventory items
        else:
            obsStr = "You have the following items in your inventory:\n"
            for obj in inventory:
                obsStr += "\t" + obj.makeDescriptionStr() + "\n"
            return obsStr

    # Performs an action in the environment, returns the result (a string observation, the reward, and whether the game is completed).
    def step_action(self, actionStr):
        self.observationStr = ""
        reward = 0

        # Check to make sure the action is in the possible actions dictionary
        if actionStr not in self.possibleActions:
            self.observationStr = "I don't understand that."
            return (self.observationStr, self.score, reward, self.gameOver, self.gameWon)

        self.numSteps += 1

        # Find the action in the possible actions dictionary
        actions = self.possibleActions[actionStr]
        action = None

        # Check for an ambiguous action (i.e. one that has multiple possible arguments)
        if (len(actions) > 1):
            # If there are multiple possible arguments, for now just choose the first one
            action = actions[0]
        else:
            # Otherwise, also just take the first action in the list of possible actions
            action = actions[0]

        # Interpret the action
        actionVerb = action[0]


        if (actionVerb == "look around"):
            # Look around the environment -- i.e. show the description of the world.
            self.observationStr = self.rootObject.makeDescriptionStr()
        elif (actionVerb == "inventory"):
            # Display the agent's inventory
            self.observationStr = self.actionInventory()

        elif (actionVerb == "examine"):
            # Examine an object
            thingToExamine = action[1]
            self.observationStr = thingToExamine.makeDescriptionStr(makeDetailed = True)
        elif (actionVerb == "open"):
            # Open a container
            thingToOpen = action[1]
            self.observationStr = self.actionOpen(thingToOpen)
        elif (actionVerb == "close"):
            # Close a container
            thingToClose = action[1]
            self.observationStr = self.actionClose(thingToClose)
        elif (actionVerb == "take"):
            # Take an object from a container
            thingToTake = action[1]
            self.observationStr = self.actionTake(thingToTake)

        elif (actionVerb == "put"):
            # Put an object in a container
            thingToMove = action[1]
            newContainer = action[2]
            self.observationStr = self.actionPut(thingToMove, newContainer)


        # Catch-all
        else:
            self.observationStr = "ERROR: Unknown action."

    def step_tick(self):
        # Do one tick of the environment
        self.doWorldTick()

    def step_calculate_score(self):
        # Calculate the score
        lastScore = self.score
        self.calculateScore()
        reward = self.score - lastScore
        return reward

    def step(self, actionStr):
        self.step_action(actionStr)
        self.step_tick()
        reward = self.step_calculate_score()
        return (self.observationStr, self.score, reward, self.gameOver, self.gameWon)


    # Call the object update for each object in the environment
    def doWorldTick(self):
        # Get a list of all objects in the environment
        allObjects = self.rootObject.getAllContainedObjectsRecursive()
        # Loop through all objects, and call their tick()
        for obj in allObjects:
            obj.tick()

    # Calculate the game score
    def calculateScore(self):
        # Baseline score
        self.score = 0

        # Check for the winning condition: all body parts with wounds have bandages on them
        allObjects = self.rootObject.getAllContainedObjectsRecursive()
        bodyPartsMissingBandages = 0
        for obj in allObjects:
            # Check if the object is an instance of the BodyPart class
            if (isinstance(obj, BodyPart)):
                # Check if the body part has a wound
                if obj.getProperty("hasWound"):
                    # Check if the body part has a bandage on it
                    bodyPartsMissingBandages += 1
                    for o in obj.getAllContainedObjectsRecursive():
                        if type(o) == Bandage:
                            self.score += 1
                            bodyPartsMissingBandages -= 1
                            break

        # If all body parts have bandages, the game is over
        if (bodyPartsMissingBandages == 0):
            self.gameOver = True
            self.gameWon = True


# Main Program
def main():

    # Create a new game
    game = TextGame(randomSeed = randomSeed)

    # Get a list of valid actions
    possibleActions = game.generatePossibleActions()
    #print("Possible actions: " + str(possibleActions.keys()))
    print("Task Description: " + game.getTaskDescription())
    print("")
    print("Initial Observation: " + game.observationStr)
    print("")
    print("Type 'help' for a list of possible actions.")
    print("")


    # Main game loop
    #while not game.gameOver:
    while True:

        # Get the player's action
        actionStr = ""
        while ((len(actionStr) == 0) or (actionStr == "help")):
            actionStr = input("> ")
            if (actionStr == "help"):
                print("Possible actions: " + str(possibleActions.keys()))
                print("")
                actionStr = ""
            elif (actionStr == "exit") or (actionStr == "quit"):
                return

        # Perform the action
        observationStr, score, reward, gameOver, gameWon = game.step(actionStr)

        # Get a list of valid actions
        possibleActions = game.generatePossibleActions()

        # Print the current game state
        print("Observation: " + observationStr)
        print("")
        print("Current step: " + str(game.numSteps))
        print("Score: " + str(score))
        print("Reward: " + str(reward))
        print("Game Over: " + str(gameOver))
        print("Game Won: " + str(gameWon))
        print("")
        print("----------------------------------------")


# Run the main program
if __name__ == "__main__":
    main()

