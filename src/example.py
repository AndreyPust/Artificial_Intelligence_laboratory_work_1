#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq
import math
from abc import ABC, abstractmethod
from collections import deque


class Problem(ABC):
    """
    Абстрактный класс для формальной постановки задачи.
    Новый домен (конкретная задача) должен специализировать этот класс,
    переопределяя методы actions и result, а при необходимости action_cost, h и is_goal.
    """

    def __init__(self, initial=None, goal=None, **kwargs):
        self.initial = initial
        self.goal = goal
        # Сохраняем все остальные переданные параметры (при желании).
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def actions(self, state):
        """
        Вернуть доступные действия (операторы) из данного состояния.
        """

        pass

    @abstractmethod
    def result(self, state, action):
        """
        Вернуть результат применения действия к состоянию.
        """

        pass

    def is_goal(self, state):
        """
        Проверка, является ли состояние целевым.
        """

        return state == self.goal

    def action_cost(self, s, a, s1):
        """
        Возвращает стоимость применения действия a,
        переводящего состояние s в состояние s1.
        По умолчанию = 1.
        """

        return 1

    def h(self, node):
        """
        Эвристическая функция; по умолчанию = 0.
        """

        return 0

    def __str__(self):
        return f"{type(self).__name__}({self.initial!r}, {self.goal!r})"


class Node:
    """
    Узел в дереве поиска.
    """

    def __init__(self, state, parent=None, action=None, path_cost=0.0):
        self.state = state  # Текущее состояние
        self.parent = parent  # Родительский узел
        self.action = action  # Действие, которое привело к этому узлу
        self.path_cost = path_cost  # Стоимость пути от начального узла

    def __repr__(self):
        return f"<Node {self.state}>"

    # Позволяет сравнивать узлы по стоимости пути (для приоритетных очередей)
    def __lt__(self, other):
        return self.path_cost < other.path_cost

    # Глубина узла — длина пути от корня (получаем рекурсивно)
    def __len__(self):
        if self.parent is None:
            return 0
        else:
            return 1 + len(self.parent)


# Специальные «сигнальные» узлы
failure = Node("failure", path_cost=math.inf)
cutoff = Node("cutoff", path_cost=math.inf)


def expand(problem, node):
    """
    Раскрываем (расширяем) узел, генерируя все дочерние узлы.
    """

    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(state=s1, parent=node, action=action, path_cost=cost)


def path_actions(node):
    """
    Последовательность действий, чтобы добраться от корня до данного узла.
    """

    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.action]


def path_states(node):
    """
    Последовательность состояний от корня до данного узла.
    """

    if node.parent is None:
        return [node.state]
    return path_states(node.parent) + [node.state]


FIFOQueue = deque  # Для поиска в ширину (очередь FIFO)
LIFOQueue = list  # Для поиска в глубину (стек LIFO)


class PriorityQueue:
    """
    Очередь с приоритетом, где элемент с минимальным значением key(item) извлекается первым.
    """

    def __init__(self, items=(), key=lambda x: x):
        self.key = key
        self.items = []  # внутри храним (priority, item)
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


def tree_search(problem, frontier):
    """
    Универсальная функция поиска по дереву.
    problem  - объект задачи (Problem).
    frontier - структура данных для хранения узлов (очередь).
               Может быть FIFOQueue (для поиска в ширину),
               LIFOQueue (для поиска в глубину) и т.д.
    Возвращает:
      - узел, если решение найдено,
      - специальный узел failure, если решения нет.
    """

    # 1. Поместить начальный узел в очередь (fringe, frontier)
    start_node = Node(problem.initial)
    frontier_append(frontier, start_node)

    # 2. Пока очередь не пуста
    while frontier_not_empty(frontier):
        # 3. Достать узел из очереди
        node = frontier_pop(frontier)

        # 4. Проверить, достигнута ли цель
        if problem.is_goal(node.state):
            return node

        # 5. Расширить узел и поместить всех потомков в очередь
        for child in expand(problem, node):
            frontier_append(frontier, child)

    # 6. Если очередь пуста, решения нет
    return failure


# Вспомогательные функции для работы с разными типами очередей:
def frontier_append(frontier, node):
    """Добавление узла в структуру данных frontier."""
    if isinstance(frontier, deque):  # FIFOQueue
        frontier.append(node)
    elif isinstance(frontier, list):  # LIFOQueue
        frontier.append(node)
    elif isinstance(frontier, PriorityQueue):
        frontier.add(node)
    else:
        raise TypeError("Неизвестный тип очереди для frontier")


def frontier_pop(frontier):
    """Извлечение узла из структуры данных frontier."""
    if isinstance(frontier, deque):  # FIFOQueue: очередь — берем слева
        return frontier.popleft()
    elif isinstance(frontier, list):  # LIFOQueue: стек — берем с конца
        return frontier.pop()
    elif isinstance(frontier, PriorityQueue):
        return frontier.pop()
    else:
        raise TypeError("Неизвестный тип очереди для frontier")


def frontier_not_empty(frontier):
    """Проверка, что очередь не пуста."""
    return len(frontier) > 0


def breadth_first_tree_search(problem):
    """
    Поиск по дереву в ширину (BFS).
    Возвращает найденный узел или специальный узел failure.
    """
    frontier = FIFOQueue()
    frontier_append(frontier, Node(problem.initial))

    while frontier:
        node = frontier.popleft()  # для FIFO очереди берём узел слева
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            frontier.append(child)

    return failure


def depth_first_tree_search(problem):
    """
    Поиск по дереву в глубину (DFS).
    Возвращает найденный узел или специальный узел failure.
    """
    frontier = LIFOQueue()
    frontier.append(Node(problem.initial))

    while frontier:
        node = frontier.pop()  # для LIFO очереди берём узел с конца
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            frontier.append(child)

    return failure
