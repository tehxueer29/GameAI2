import math
import pickle
from constants import *
from run import GameController 

class State:
    def __init__(self, p1):
        self.state = []
        self.p1 = p1
        self.isEnd = False
        self.finalScore = 0
    
    def availableDirections(self, pacman):
        return pacman.validDirections()
    
    def manhattanDistance(pos1, pos2):
        """
        Returns the Manhattan distance between two points.

        The Manhattan distance is the distance between two points measured along
        axes at right angles. In a plane with p1 at (x1, y1) and p2 at (x2, y2),
        it is |x1 - x2| + |y1 - y2|.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # Returns the direction of the closest ghost relative to pacman
    # if the ghost is within a certain range. Else, returns None.
    def getClosestGhostDirection(self, ghosts, pacman_target):
        closest_ghost = None
        closest_distance = 0
        for ghost in ghosts:
            # manhatten distance
            distance = abs(pacman_target[0] - ghost.position.x) + abs(pacman_target[1] - ghost.position.y)
            hunt_distance = abs(pacman_target[0] - ghost.goal.x) + abs(pacman_target[1] - ghost.goal.y)
            distance = int(distance)
            # distance = math.sqrt((pacman_target[0] - ghost.position.x)**2 + (pacman_target[1] - ghost.position.y)**2)
            if closest_ghost is None or distance < closest_distance:
                closest_ghost = ghost
                closest_distance = distance
        
        # can modify this. closest ghost within 80 units of pacman's range
        # also checking whr the ghost is in a btr way, currently it only checks if the ghost is 
        # to a certain direction (e.g. if ghost is to the left, it wont know if ghost is upper or lower
        # left) so it returns  the direction rather than the distance. (shd return both i think)
        # print(closest_distance)
        # if closest_distance <= 200:
        vec = (closest_ghost.position.x - pacman_target[0], closest_ghost.position.y - pacman_target[1])
        # print("distance" + str(distance))

        if abs(vec[1]) >= abs(vec[0]): 
            if vec[1] >= 0:
                return DOWN, distance, hunt_distance
            else:
                return UP, distance, hunt_distance
        else: 
            if vec[0] >= 0:
                return RIGHT, distance, hunt_distance
            else:
                return LEFT, distance, hunt_distance
        # else: 
        #     return None, 0

    # xe check if any ghost is in freight mode
    def isFreightMode(self, ghosts):
        for ghost in ghosts:
            if ghost.mode.current == FREIGHT:
                return True
        return False
    
    # xe get closest pellet distance
    def getClosestPelletDistance(self, game, pacman_target):
        pellets = game.pellets.pelletList
        distances = []
        for pellet in pellets:
            distance = abs(pacman_target[0] - pellet.position.x) + abs(pacman_target[1] - pellet.position.y)
            distances.append(distance)
        return min(distances)

    # Updates the state with the current game world's information.
    def updateState(self, ghosts, pacman_target, game):
        is_dead = not game.pacman.alive
        is_freight = self.isFreightMode(ghosts)
        closest_pellet_dist = self.getClosestPelletDistance(game, pacman_target)
        closest_ghost, distance, hunt_distance = self.getClosestGhostDirection(ghosts, pacman_target)
        self.state = [int(pacman_target[0]), int(pacman_target[1]), closest_ghost, is_dead, is_freight, distance, closest_pellet_dist, hunt_distance]
    
    # Apply the chosen action (direction) to the game.
    def applyAction(self, game, direction):
        game.pacman.learntDirection = direction
        game.update()
    
    # Checks if game is over i.e. level completed or all lives lost.
    def gameEnded(self, game):
        if game.lives <= 0 :
            self.isEnd = True
            self.finalScore = game.score
            return 0
        if game.level > self.level:
            return 1
        else:
            return None
    
    # Checks if game is paused i.e. after one life is lost or at the
    # beginning of new game. If it is, resumes it.
    def gamePaused(self, game):
        if game.pause.paused:
            if game.pacman.alive:
                game.pause.setPause(playerPaused=True)
                if not game.pause.paused:
                    game.textgroup.hideText()
                    game.showEntities()

    # Main method for training.
    def play(self, iterations=100):
        for i in range(iterations):
            # if i % 1000 == 0:
            #     print("Iterations {}".format(i))
            if i % 25 == 0:
                p1.savePolicy()
                print("policy saved!")
            print("iteration" + str(i))
            game = GameController()
            game.startGame()
            game.update()
            pacman_target = game.nodes.getPixelsFromNode(game.pacman.target)
            self.updateState(game.ghosts, pacman_target, game)
            self.level = game.level
            while not self.isEnd:
                possible_directions = self.availableDirections(game.pacman)
                p1_action = self.p1.getAction(self.state, possible_directions, game.score)
                # take action and update board state
                self.applyAction(game, p1_action)
                pacman_target = game.nodes.getPixelsFromNode(game.pacman.target)
                self.updateState(game.ghosts, pacman_target, game)

                # check board status if it is end
                self.gamePaused(game)
                result = self.gameEnded(game)
                if result is not None:
                    self.p1.final(self.state, game.score)
                    game.restartGame()
                    del game
                    self.isEnd = False
                    break

                else:
                    # next frame iteration
                    continue



if __name__ == "__main__":
    #### PARAMETERS:
    # ALPHA -> Learning Rate
    # controls how much influence the current feedback value has over the stored Q-value.

    # GAMMA -> Discount Rate
    # how much an action’s Q-value depends on the Q-value at the state (or states) it leads to.

    # RHO -> Randomness of Exploration
    # how often the algorithm will take a random action, rather than the best action it knows so far.

    # NU: The Length of Walk
    # number of iterations that will be carried out in a sequence of connected actions.
    
    exploration_rho=0.2
    lr_alpha= 0.2
    discount_rate_gamma=0.9
    walk_len_nu = 0.2

    # training
    from player import *
    p1 = Player("p1", exploration_rho, lr_alpha, discount_rate_gamma, walk_len_nu)
    st = State(p1)

    # # # TRAINING
    print("Training...")
    p1.loadPolicy("trained_controller")
    st.play(10000)
    p1.savePolicy()

    # DEMO
    # demo_p1 = Player("demo", exploration_rho=0, lr_alpha=0)
    # demo_p1.loadPolicy("trained_controller")
    # stDemo = State(demo_p1)
    # stDemo.play()