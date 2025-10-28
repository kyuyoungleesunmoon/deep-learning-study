#!/usr/bin/env python3
"""
Complete Module A & B Generator
Creates comprehensive AI textbook notebooks
"""
import json
import os

def md(t):
    return {"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in t.strip().split("\n")]}

def code(t):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[l+"\n" for l in t.strip().split("\n")]}

def save_nb(cells, path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"✅ {path}: {len(cells)} cells, {os.path.getsize(path)/1024:.1f}KB")

if __name__ == "__main__":
    print("Generating comprehensive AI textbook modules...")
    print("=" * 70)
    
    # Module A and B will be imported and generated
    # This is the main entry point
