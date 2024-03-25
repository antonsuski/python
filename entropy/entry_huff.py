import math
import sys

def calculate_symbol_counts(buffer):
    symbol_counts = {}
    
    # Подсчитываем частоту встречаемости каждого символа в буфере
    for symbol in buffer:
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    
    return symbol_counts

def calculate_entropy_and_average_code_length(symbol_counts, total_symbols):
    # Вычисляем энтропию данных
    entropy = 0
    for count in symbol_counts.values():
        probability = count / total_symbols
        entropy -= probability * math.log2(probability)
    
    # Вычисляем среднюю длину кода
    average_code_length = 0
    for count in symbol_counts.values():
        probability = count / total_symbols
        average_code_length += probability * math.ceil(math.log2(1 / probability))
    
    return entropy, average_code_length

class Node:
    def __init__(self, symbol=None, frequency=0):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

def build_huffman_tree(symbol_counts):
    # Создаем список узлов для каждого символа с его частотой
    nodes = [Node(symbol, count) for symbol, count in symbol_counts.items()]
    
    # Строим дерево Хаффмана
    while len(nodes) > 1:
        # Сортируем узлы по частоте
        nodes = sorted(nodes, key=lambda x: x.frequency)
        
        # Получаем два узла с наименьшей частотой
        left = nodes.pop(0)
        right = nodes.pop(0)
        
        # Создаем новый узел, суммируя частоты дочерних узлов
        new_node = Node(frequency=left.frequency + right.frequency)
        new_node.left = left
        new_node.right = right
        
        # Добавляем новый узел в список узлов
        nodes.append(new_node)
    
    # Возвращаем корень дерева
    return nodes[0]

def huffman_codes(node, prefix="", codes={}):
    # Если узел - лист, добавляем его символ и код в словарь
    if node.symbol is not None:
        codes[node.symbol] = prefix
    else:
        # Рекурсивно обходим левое и правое поддеревья
        huffman_codes(node.left, prefix + "0", codes)
        huffman_codes(node.right, prefix + "1", codes)
    
    return codes

def huffman_encode(buffer, huffman_codes):
    encoded_buffer = ""
    
    # Проходим по каждому символу в буфере и добавляем его код Хаффмана к закодированному буферу
    for symbol in buffer:
        encoded_buffer += huffman_codes[symbol]
    
    return encoded_buffer

def huffman_decode(encoded_buffer, huffman_tree):
    decoded_buffer = ""
    current_node = huffman_tree
    
    # Проходим по каждому биту в закодированном буфере
    for bit in encoded_buffer:
        # Если бит - 0, идем влево по дереву
        if bit == "0":
            current_node = current_node.left
        # Если бит - 1, идем вправо по дереву
        elif bit == "1":
            current_node = current_node.right
        
        # Если мы достигли листа, добавляем его символ в декодированный буфер
        if current_node.symbol is not None:
            decoded_buffer += chr(current_node.symbol)  # Преобразуем байт в символ
            current_node = huffman_tree  # Возвращаемся к корню дерева
    
    return decoded_buffer

def huffman_decode(encoded_buffer, huffman_tree):
    decoded_buffer = ""
    current_node = huffman_tree
    
    # Проходим по каждому биту в закодированном буфере
    for bit in encoded_buffer:
        # Если бит - 0, идем влево по дереву
        if bit == "0":
            current_node = current_node.left
        # Если бит - 1, идем вправо по дереву
        elif bit == "1":
            current_node = current_node.right
        
        # Если мы достигли листа, добавляем его символ в декодированный буфер
        if current_node.symbol is not None:
            decoded_buffer += current_node.symbol
            current_node = huffman_tree  # Возвращаемся к корню дерева
    
    return decoded_buffer

if len(sys.argv) == 2:
    file_path = sys.argv[1]

    # Считываем данные из файла
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    
    # Подсчитываем количество символов в файле
    total_symbols = len(data)
    # Создаем список уникальных значений
    uniq_symbols_list = calculate_symbol_counts(data)

    # Делаем расчет энтрапии
    entropy, avg_code_length = calculate_entropy_and_average_code_length(uniq_symbols_list,total_symbols)
    print("Энтрапия данных:", entropy)
    print("Средняя длина кода:", avg_code_length)

    # Создаем дерево Хаффмана и коды
    huffman_tree = build_huffman_tree(uniq_symbols_list)
    huffman_codes = huffman_codes(huffman_tree)
    print("Хаффман коды для каждого символа:")
    for symbol, code in huffman_codes.items():
        print(f"Символ '{symbol}': Код {code}")

    # Кодируем буфер
    encoded_buffer = huffman_encode(data, huffman_codes)
    print("Закодированный буфер:")
    print(encoded_buffer)

    # Декодируем буфер
    decoded_buffer = huffman_decode(encoded_buffer, huffman_tree)
    print("Декодированный буфер:")
    print(decoded_buffer)

else:
    print("Не правильное использование. Пример:\n\t python entry_huff.py [path]")
