#!/usr/bin/env python3
"""
Автоматическое исправление SageAttention3 для компиляции на Windows с MSVC
Использование: python fix_sageattention3.py <путь_к_sageattention3_blackwell>
Пример: python fix_sageattention3.py C:\\Users\\gta4r\\Desktop\\ComfyUI\\ComfyUI_windows_portable\\SageAttention\\sageattention3_blackwell
"""

import os
import sys
import re
from pathlib import Path

def fix_kernel_traits(file_path):
    """Добавляет typename перед LayoutSF и SfAtom в kernel_traits.h"""
    print(f"[*] Исправление {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Резервная копия
    backup_path = str(file_path) + '.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[+] Создана резервная копия: {backup_path}")

    # Исправление 1: добавить typename перед LayoutSF
    content = re.sub(
        r'(\s+)using\s+LayoutSF\s*=\s*SharedStorageQKVO::LayoutSF;',
        r'\1using LayoutSF = typename SharedStorageQKVO::LayoutSF;',
        content
    )

    # Исправление 2: добавить typename перед SfAtom
    content = re.sub(
        r'(\s+)using\s+SfAtom\s*=\s*SharedStorageQKVO::SfAtom;',
        r'\1using SfAtom = typename SharedStorageQKVO::SfAtom;',
        content
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[+] Файл {file_path} исправлен успешно!")
    return True

def fix_setup_py(file_path):
    """Добавляет флаги MSVC в setup.py"""
    print(f"[*] Исправление {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Резервная копия
    backup_path = str(file_path) + '.backup'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[+] Создана резервная копия: {backup_path}")

    # Проверяем, не добавлены ли уже флаги
    if '/Zc:__cplusplus' in content or 'Zc:__cplusplus' in content:
        print("[!] Похоже, что флаги MSVC уже добавлены. Пропускаем...")
        return True

    # Ищем секцию extra_compile_args
    # Добавляем import os в начало файла, если его нет
    if 'import os' not in content:
        content = 'import os\n' + content
        print("[+] Добавлен 'import os'")

    # Добавляем код для флагов MSVC после определения extra_compile_args
    msvc_flags_code = '''
# Fix for Windows/MSVC compilation
if os.name == 'nt':  # Windows
    extra_compile_args["cxx"].extend([
        "/Zc:__cplusplus",  # Correct C++ standard macro definition
        "/bigobj",          # Support for larger object files
        "/permissive-"      # Strict C++ standard conformance
    ])
    extra_compile_args["nvcc"].extend([
        "-Xcompiler", "/Zc:__cplusplus",
        "-Xcompiler", "/bigobj",
        "-Xcompiler", "/permissive-"
    ])
'''

    # Ищем место после определения extra_compile_args
    # Обычно это после закрывающей фигурной скобки словаря
    pattern = r'(extra_compile_args\s*=\s*\{[^}]+\})'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        insert_position = match.end()
        content = content[:insert_position] + '\n' + msvc_flags_code + content[insert_position:]
        print("[+] Добавлены флаги MSVC в extra_compile_args")
    else:
        print("[!] ВНИМАНИЕ: Не найдена секция extra_compile_args. Добавляем код в конец файла перед setup()...")
        # Ищем вызов setup() и добавляем перед ним
        setup_match = re.search(r'(\nsetup\s*\()', content)
        if setup_match:
            insert_position = setup_match.start()
            content = content[:insert_position] + '\n' + msvc_flags_code + '\n' + content[insert_position:]
        else:
            print("[!] ОШИБКА: Не найден вызов setup(). Требуется ручное исправление.")
            return False

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[+] Файл {file_path} исправлен успешно!")
    return True

def main():
    if len(sys.argv) < 2:
        print("Использование: python fix_sageattention3.py <путь_к_sageattention3_blackwell>")
        print("Пример: python fix_sageattention3.py C:\\path\\to\\sageattention3_blackwell")
        sys.exit(1)

    base_path = Path(sys.argv[1])

    if not base_path.exists():
        print(f"[!] ОШИБКА: Путь не существует: {base_path}")
        sys.exit(1)

    print(f"[*] Базовый путь: {base_path}")
    print("[*] Начинаем исправление файлов...\n")

    # Исправление kernel_traits.h
    kernel_traits_path = base_path / "sageattn3" / "blackwell" / "kernel_traits.h"
    if not kernel_traits_path.exists():
        print(f"[!] ОШИБКА: Не найден файл: {kernel_traits_path}")
        sys.exit(1)

    success1 = fix_kernel_traits(kernel_traits_path)
    print()

    # Исправление setup.py
    setup_py_path = base_path / "setup.py"
    if not setup_py_path.exists():
        print(f"[!] ОШИБКА: Не найден файл: {setup_py_path}")
        sys.exit(1)

    success2 = fix_setup_py(setup_py_path)
    print()

    if success1 and success2:
        print("[✓] ВСЕ ИСПРАВЛЕНИЯ ПРИМЕНЕНЫ УСПЕШНО!")
        print("\nТеперь можно запустить компиляцию:")
        print(f"cd {base_path}")
        print("python setup.py build_ext --inplace")
    else:
        print("[✗] НЕКОТОРЫЕ ИСПРАВЛЕНИЯ НЕ УДАЛИСЬ. Проверьте вывод выше.")
        sys.exit(1)

if __name__ == "__main__":
    main()
