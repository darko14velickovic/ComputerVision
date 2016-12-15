
class Node:

    def __init__(self):
        self.bias = 0
        self.value = 0
        self.connections = []  # first hidden layer connected to Input
        self.weights = []

    def propagate(self):
        for i in range(0, self.connections.__len__()):
            self.connections[i].value = self.value * self.weights[i]

    def add_connection(self, node, weight):
        self.connections.append(node)
        self.weights.append(weight)

    def show(self):
        return "Value: "+ str(self.value) +", Bias: "+ str(self.bias)

class Network:

    def __init__(self):
        self.dict = {}

    def add_node(self, name, node):
        self.dict[name] = node

    def show_network(self):
        for key, value in self.dict.iteritems():
            print key + value.show()

    def propagate(self):
        self.dict["layer1node0"].propagate()
