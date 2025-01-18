#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from itertools import permutations


class Problem:
    """
    Формальное описание задачи, задаем узлы графа и связи между ними.
    """

    def __init__(self, initial=None, goal=None, **kwds):
        self.__dict__.update(initial=initial, goal=goal, **kwds)

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def is_goal(self, state):
        return state == self.goal

    def action_cost(self, s, a, s1):
        return 1

    def h(self, node):
        return 0

    def __str__(self):
        return "{}({!r}, {!r})".format(type(self).__name__, self.initial, self.goal)


class Node:
    def __init__(self, state, parent=None, action=None, path_cost=float(0)):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __repr__(self):
        return "<{}>".format(self.state)

    def __len__(self):
        return 0 if self.parent is None else (1 + len(self.parent))

    def __lt__(self, other):
        return self.path_cost < other.path_cost


failure = Node("failure", path_cost=math.inf)
cutoff = Node("cutoff", path_cost=math.inf)


def expand(problem, node):
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)


def path_actions(node):
    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.action]


def path_states(node):
    if node in (cutoff, failure, None):
        return []
    return path_states(node.parent) + [node.state]


class TSPProblem(Problem):
    """
    Наследует Problem, задавая методы для действий, переходов и стоимости на основе структуры графа.
    """

    def __init__(self, initial, goal, graph):
        super().__init__(initial=initial, goal=goal)
        self.graph = graph

    def actions(self, state):
        return [neighbor for neighbor in self.graph[state]]

    def result(self, state, action):
        return action

    def action_cost(self, s, a, s1):
        return self.graph[s][s1]


def solve_tsp(problem):
    """
    Функция реализующая метод полного перебора задачи коммивояжера.
    :param problem: задача.
    :return: список городов по порядку и длина пути.
    """

    nodes = list(problem.graph.keys())
    nodes.remove(problem.initial)

    # Зададим изначально максимальную длину пути
    min_path_cost = math.inf

    # Зададим вначале пустой путь
    min_path = None

    for perm in permutations(nodes):
        path = [problem.initial] + list(perm) + [problem.initial]
        cost = sum(problem.action_cost(path[i], None, path[i + 1]) for i in range(len(path) - 1))

        if cost < min_path_cost:
            min_path_cost = cost
            min_path = path

    return min_path, min_path_cost


if __name__ == "__main__":
    # Задаем граф городов Австралии в виде списка смежности
    graph_australia = {
        "Буриндал": {
            "Уоррен": 271,
            "Нинган": 156,
            "Кобар": 204,
            "Нарромин": 41,
            "Гилгандра": float("inf"),
            "Канбелего": float("inf"),
            "Наймаджи": float("inf"),
            "Гулгуния": float("inf"),
        },
        "Уоррен": {
            "Буриндал": 271,
            "Гилгандра": 103,
            "Нарромин": 86,
            "Нинган": 78,
            "Кобар": float("inf"),
            "Канбелего": float("inf"),
            "Наймаджи": float("inf"),
            "Гулгуния": float("inf"),
        },
        "Нинган": {
            "Буриндал": 156,
            "Кобар": 86,
            "Наймаджи": 122,
            "Уоррен": 78,
            "Канбелего": 86,
            "Гилгандра": 99,
            "Нарромин": float("inf"),
            "Гулгуния": float("inf"),
        },
        "Кобар": {
            "Буриндал": 204,
            "Канбелего": 50,
            "Наймаджи": 97.5,
            "Гулгуния": 110,
            "Нинган": 86,
            "Гилгандра": float("inf"),
            "Нарромин": float("inf"),
            "Уоррен": float("inf"),
        },
        "Канбелего": {
            "Кобар": 50,
            "Наймаджи": 61,
            "Нинган": 86,
            "Буриндал": float("inf"),
            "Гилгандра": float("inf"),
            "Нарромин": float("inf"),
            "Уоррен": float("inf"),
            "Гулгуния": float("inf"),
        },
        "Наймаджи": {
            "Кобар": 97.5,
            "Нинган": 122,
            "Гулгуния": 46,
            "Канбелего": 61,
            "Буриндал": float("inf"),
            "Гилгандра": float("inf"),
            "Нарромин": float("inf"),
            "Уоррен": float("inf"),
        },
        "Гулгуния": {
            "Кобар": 110,
            "Наймаджи": 46,
            "Буриндал": float("inf"),
            "Гилгандра": float("inf"),
            "Нарромин": float("inf"),
            "Уоррен": float("inf"),
            "Канбелего": float("inf"),
            "Нинган": float("inf"),
        },
        "Гилгандра": {
            "Нарромин": 100,
            "Уоррен": 103,
            "Нинган": 99,
            "Буриндал": float("inf"),
            "Канбелего": float("inf"),
            "Наймаджи": float("inf"),
            "Гулгуния": float("inf"),
            "Кобар": float("inf"),
        },
        "Нарромин": {
            "Буриндал": 41,
            "Уоррен": 86,
            "Гилгандра": 100,
            "Канбелего": float("inf"),
            "Наймаджи": float("inf"),
            "Гулгуния": float("inf"),
            "Нинган": float("inf"),
            "Кобар": float("inf"),
        },
    }

    # Создаем объект задачи и находим решение
    tsp_problem = TSPProblem("Буриндал", "Буриндал", graph_australia)
    solution_path, solution_cost = solve_tsp(tsp_problem)

    print("Минимальный путь для задачи коммивояжера (минимальный Гамильтонов цикл):", solution_path)
    print("Стоимость минимального пути:", solution_cost)
