import numpy as np

class BayesianNetwork:
    def __init__(self, filename):
        f = open(filename, 'r') 
        N = int(f.readline())
        lines = f.readlines()
        graph = {}
        graph['node'] = []
        graph['edge'] = {}
        temp_lines = []
        self.valueHash = {}

        for line in lines:
            node, parents, domain, shape, probabilities = self.__extract_model(line)
            temp_lines.append((node, parents, domain, shape, probabilities))
            # YOUR CODE HERE

            # Constructing graph
            graph['node'].append(node)

            if node not in graph['edge']:
                graph['edge'][node] = []

            for parent in parents:
                if parent not in graph['edge']:
                    graph['edge'][parent] = []
                graph['edge'][parent].append(node)

            # Mapping string-based value to index
            self.valueHash[node] = {k: i for i, k in enumerate(domain)}

        # Create topological order
        self.node_order = self.topologicalSort(graph)
        self.node_order_dict = {k: i for i, k in enumerate(self.node_order)}
        #print(node_order)

        self.lines = {}
        self.factors = []
        # Reorder cpt
        for node, parents, domain, shape, probabilities in temp_lines:
            #print('------')
            #print(node, parents, domain, shape, probabilities)
            
            parents_dict = {k: i for i, k in enumerate(parents)}
            new_order = []
            new_parents = []
            for i, parent in enumerate(self.node_order):
                if parent in parents_dict:
                    new_order.append(parents_dict[parent])
                    new_parents.append(parent)

            new_order.append(-1)
            new_order = tuple(new_order)
            new_probabilities = probabilities.transpose(new_order)

            for i, n in enumerate(self.node_order):
                if n not in new_parents and n is not node:
                    new_probabilities = np.expand_dims(new_probabilities, axis=i)

            self.lines[node] = {'parents': new_parents, 'domain': domain, 'probabilities': new_probabilities}
            #self.lines.append((node, new_parents, domain, new_probabilities.shape, new_probabilities))
            self.factors.append(new_probabilities.astype(np.float64))
            #print(node, new_parents, domain, new_probabilities.shape, new_probabilities)
        
        f.close()

    def topologicalSortUtil(self, v, visited, stack, graph):
        visited[v] = True
 
        for u in graph['edge'][v]:
            if u not in visited:
                self.topologicalSortUtil(u, visited, stack, graph)

        stack.append(v)
 
    def topologicalSort(self, graph):
        visited = {}
        stack = []
 
        for v in graph['node']:
            if v not in visited:
                self.topologicalSortUtil(v, visited, stack, graph)

        return stack[::-1]

    def exact_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())
        # YOUR CODE HERE
        #print(query_variables, evidence_variables)

        current_factors = [_ for _ in self.factors]
        joint_factor = 1
        for factor in current_factors:
            joint_factor = joint_factor * factor

        for node in self.node_order:
            if node not in query_variables and node not in evidence_variables:
                joint_factor = self.marginalize(joint_factor, dim=self.node_order_dict[node])

        for e in evidence_variables:
            dim = self.node_order_dict[e]
            value = self.valueHash[e][evidence_variables[e]]
            joint_factor = self.select(joint_factor, dim, value)

        joint_factor = joint_factor / np.sum(joint_factor)
        for q in query_variables:
            dim = self.node_order_dict[q]
            value = self.valueHash[q][query_variables[q]]
            joint_factor = self.select(joint_factor, dim, value)

        result = joint_factor.item()

        f.close()
        return result

    def exact_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())
        # YOUR CODE HERE
        current_factors = [_ for _ in self.factors]

        for node in self.node_order:
            if node not in query_variables and node not in evidence_variables:
                current_factors = self.eliminate(current_factors, node)

        joint_factor = 1
        for factor in current_factors:
            joint_factor = joint_factor * factor

        for e in evidence_variables:
            dim = self.node_order_dict[e]
            value = self.valueHash[e][evidence_variables[e]]
            joint_factor = self.select(joint_factor, dim, value)

        joint_factor = joint_factor / np.sum(joint_factor)
        for q in query_variables:
            dim = self.node_order_dict[q]
            value = self.valueHash[q][query_variables[q]]
            joint_factor = self.select(joint_factor, dim, value)

        result = joint_factor.item()

        f.close()
        return result

    def eliminate(self, factors, node):
        relevant_factors = []
        new_factors = []
        for factor in factors:
            if factor.shape[self.node_order_dict[node]] != 1:
                relevant_factors.append(factor)
            else:
                new_factors.append(factor)

        if len(relevant_factors):
            new_factor = 1
            for factor in relevant_factors:
                new_factor = new_factor * factor
        
        new_factor = self.marginalize(new_factor, dim=self.node_order_dict[node])
        new_factors.append(new_factor)
        return new_factors

    def marginalize(self, factor, dim=-1):
        factor = np.sum(factor, axis=dim, keepdims=True)
        return factor

    def select(self, factor, dim=-1, value=0):
        factor = np.swapaxes(factor, dim, 0)
        factor = factor[value:value+1]
        factor = np.swapaxes(factor, dim, 0)
        return factor
    
    """
    #def approx_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())
        # YOUR CODE HERE
        n = 0
        sample_dict = {}
        for _ in range(1000):
            sample = self.joint_sampling(query_variables, evidence_variables)
            if sample is None:
                continue

            if sample not in sample_dict:
                sample_dict[sample] = 1
            else:
                sample_dict[sample] += 1
            n += 1

        if n == 0:
            result = 0
        else:
            n_ps = 0
            for sample in sample_dict:
                ret = sample_dict[sample]
                for node in query_variables:
                    if sample[self.node_order_dict[node]] != self.valueHash[node][query_variables[node]]:
                        ret = 0
                        break
                n_ps += ret

            result = n_ps / n

        f.close()
        return result

    #def joint_sampling(self, query_variables, evidence_variables):
        sampled_node = {}
        for node in self.node_order:
            result = self.node_sampling(node, sampled_node)
            sampled_node[node] = result

        for node in evidence_variables:
            if sampled_node[node] != self.valueHash[node][evidence_variables[node]]:
                return None

        sample = [sampled_node[_] for _ in self.node_order]
        return tuple(sample)

    #def node_sampling(self, node, sampled_node):
        factor = self.lines[node]['probabilities']
        for n in sampled_node:
            if n in self.lines[node]['parents']:
                dim = self.node_order_dict[n]
                value = sampled_node[n]
                factor = self.select(factor, dim, value)
        
        factor = factor.squeeze()
        factor = factor / np.sum(factor)
        factor = factor.reshape(-1)
        result = np.random.choice(np.arange(factor.shape[0]), replace=True, p=factor)
        return result

    """
    
    def approx_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        query_variables, evidence_variables = self.__extract_query(f.readline())
        # YOUR CODE HERE
        n = 0
        sample_dict = {}
        for _ in range(1000):
            sample, w = self.joint_sampling(query_variables, evidence_variables)
            if sample not in sample_dict:
                sample_dict[sample] = w
            else:
                sample_dict[sample] += w
            n += w

        if n == 0:
            result = 0
        else:
            n_ps = 0
            for sample in sample_dict:
                ret = sample_dict[sample]
                for node in query_variables:
                    if sample[self.node_order_dict[node]] != self.valueHash[node][query_variables[node]]:
                        ret = 0
                        break
                n_ps += ret

            result = n_ps / n

        f.close()
        return result

    def joint_sampling(self, query_variables, evidence_variables):
        sampled_node = {}
        w = 1.0
        for node in self.node_order:
            if node in evidence_variables:
                result = self.valueHash[node][evidence_variables[node]]
                factor = self.lines[node]['probabilities']
                for parent in self.lines[node]['parents']:
                    dim = self.node_order_dict[parent]
                    value = sampled_node[parent]
                    factor = self.select(factor, dim, value)

                factor = factor / np.sum(factor)
                dim = self.node_order_dict[node]
                value = result
                factor = self.select(factor, dim, value)
                w *= factor.item()
            else:
                result = self.node_sampling(node, sampled_node)

            sampled_node[node] = result

        sample = [sampled_node[_] for _ in self.node_order]
        return tuple(sample), w

    def node_sampling(self, node, sampled_node):
        factor = self.lines[node]['probabilities']
        for n in sampled_node:
            if n in self.lines[node]['parents']:
                dim = self.node_order_dict[n]
                value = sampled_node[n]
                factor = self.select(factor, dim, value)
        
        factor = factor.squeeze()
        factor = factor / np.sum(factor)
        factor = factor.reshape(-1)
        result = np.random.choice(np.arange(factor.shape[0]), replace=True, p=factor)
        return result

    def __extract_model(self, line):
        parts = line.split(';')
        node = parts[0]
        if parts[1] == '':
            parents = []
        else:
            parents = parts[1].split(',')
        domain = parts[2].split(',')
        shape = eval(parts[3])
        probabilities = np.array(eval(parts[4])).reshape(shape)
        return node, parents, domain, shape, probabilities

    def __extract_query(self, line):
        parts = line.split(';')

        # extract query variables
        query_variables = {}
        for item in parts[0].split(','):
            if item is None or item == '':
                continue
            lst = item.split('=')
            query_variables[lst[0]] = lst[1]

        # extract evidence variables
        evidence_variables = {}
        for item in parts[1].split(','):
            if item is None or item == '':
                continue
            lst = item.split('=')
            evidence_variables[lst[0]] = lst[1]
        return query_variables, evidence_variables
