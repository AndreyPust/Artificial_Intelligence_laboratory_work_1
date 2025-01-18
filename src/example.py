import heapq
import math
from abc import ABC, abstractmethod
from collections import deque


# -------------------------------------------------------------
# Абстрактный класс Problem
# -------------------------------------------------------------
class Problem(ABC):
    """Абстрактный класс для формальной постановки задачи."""

    def __init__(self, initial=None, goal=None, **kwargs):
        self.initial = initial
        self.goal = goal
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def actions(self, state):
        """Должна вернуть доступные действия (операторы) из данного состояния."""
        pass

    @abstractmethod
    def result(self, state, action):
        """Результат применения действия к состоянию."""
        pass

    def is_goal(self, state):
        """Определение, является ли состояние конечным."""
        return state == self.goal

    def action_cost(self, s, a, s1):
        """Стоимость шага (s->s1) под действием a.
        По умолчанию = 1, но в реальных задачах переопределяется."""
        return 1

    def h(self, node):
        """Эвристическая функция (для A*), по умолчанию 0."""
        return 0


# -------------------------------------------------------------
# Класс Node
# -------------------------------------------------------------
class Node:
    """Узел в дереве поиска."""

    def __init__(self, state, parent=None, action=None, path_cost=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __repr__(self):
        return f"<Node {self.state}>"

    def __lt__(self, other):
        """Сравнение по path_cost (чтобы работать в приоритетных очередях)."""
        return self.path_cost < other.path_cost

    def __len__(self):
        """Глубина узла в дереве — расстояние до корня."""
        return 0 if self.parent is None else (1 + len(self.parent))


failure = Node("failure", path_cost=math.inf)
cutoff = Node("cutoff", path_cost=math.inf)


# -------------------------------------------------------------
# Вспомогательные функции
# -------------------------------------------------------------
def path_actions(node):
    """Последовательность действий для достижения данного узла."""
    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.action]


def path_states(node):
    """Последовательность состояний для достижения данного узла."""
    if node.parent is None:
        return [node.state]
    else:
        return path_states(node.parent) + [node.state]


def expand(problem, node):
    """Раскрываем (генерируем) дочерние узлы для 'node'."""
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, parent=node, action=action, path_cost=cost)


# -------------------------------------------------------------
# Очереди: FIFO, LIFO, PriorityQueue
# -------------------------------------------------------------
FIFOQueue = deque  # поиск в ширину
LIFOQueue = list  # поиск в глубину


class PriorityQueue:
    """Очередь с приоритетом, извлекает элемент с минимальным key(item)."""

    def __init__(self, items=(), key=lambda x: x):
        self.key = key
        self.items = []
        for item in items:
            self.add(item)

    def add(self, item):
        heapq.heappush(self.items, (self.key(item), item))

    def pop(self):
        return heapq.heappop(self.items)[1]

    def top(self):
        return self.items[0][1]

    def __len__(self):
        return len(self.items)


# -------------------------------------------------------------
# Универсальный поиск по дереву (tree_search),
# но мы сделаем специализацию под Uniform Cost Search
# -------------------------------------------------------------
def uniform_cost_tree_search(problem):
    """
    Поиск по дереву, где выбор узла происходит на основе
    наименьшего path_cost.
    """
    node = Node(problem.initial)
    if problem.is_goal(node.state):
        return node

    frontier = PriorityQueue(key=lambda n: n.path_cost)
    frontier.add(node)

    while len(frontier) > 0:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            frontier.add(child)

    return failure


class MapProblem(Problem):
    def __init__(self, initial, goal, graph):
        super().__init__(initial=initial, goal=goal)
        self.graph = graph

    def actions(self, state):
        """Возвращаем все соседние города, куда можно поехать из 'state'."""
        return list(self.graph[state].keys())

    def result(self, state, action):
        """
        Результатом перехода (действия) 'action' из 'state' будет сам город 'action'.
        Здесь 'action' — это название соседнего города.
        """
        return action

    def action_cost(self, s, a, s1):
        """
        Стоимость пути = вес дуги в графе.
        s - исходный город,
        a - следующий город (действие),
        s1 - тоже следующий город (по сути a == s1).
        """
        return self.graph[s][s1]


def main():
    """Главна функция программы"""
    # Задаем пример укороченного графа для поиска (граф городов Австралии)
    graph = {
        "Буриндал": {"Уоррен": 271, "Нинган": 156, "Кобар": 204, "Нарромин": 41},
        "Уоррен": {"Буриндал": 271, "Гилгандра": 103, "Нарромин": 86, "Нинган": 78},
        "Нарромин": {"Буриндал": 41, "Гилгандра": 100, "Уоррен": 86},
        "Гилгандра": {"Нарромин": 100, "Нинган": 99, "Уоррен": 103},
        "Нинган": {"Буриндал": 156, "Гилгандра": 99, "Уоррен": 78, "Кобар": 86, "Наймаджи": 122, "Канбелего": 86},
        "Кобар": {"Буриндал": 204, "Нинган": 86, "Канбелего": 50, "Наймаджи": 97.5, "Гулгуния": 110},
        "Канбелего": {"Нинган": 86, "Кобар": 50, "Наймаджи": 61},
        "Наймаджи": {"Нинган": 122, "Кобар": 97.5, "Канбелего": 61, "Гулгуния": 46},
        "Гулгуния": {"Кобар": 110, "Наймаджи": 46},
    }

    # Создаём задачу для поиска кратчайшего пути от Буриндала до Гулгунии
    problem = MapProblem(initial="Буриндал", goal="Гулгуния", graph=graph)

    # Вызываем поиск по дереву (по наименьшей стоимости)
    solution_node = uniform_cost_tree_search(problem)

    if solution_node is failure:
        print("Решение не найдено!")
    else:
        # Восстанавливаем путь (последовательность городов) для вывода
        path = path_states(solution_node)
        cost = solution_node.path_cost
        print("Найден путь:", " -> ".join(path))
        print("Общая стоимость (расстояние) пути км.:", cost)


if __name__ == "__main__":
    main()
