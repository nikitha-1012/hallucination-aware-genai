\# Python Fundamentals



\## Variables and Data Types



A variable in Python is a name that refers to a value. Python variables do not require an explicit type declaration.



Common built-in data types include:

\- int for integers

\- float for decimal numbers

\- str for text

\- bool for True or False values

\- list for ordered mutable collections

\- tuple for ordered immutable collections

\- set for unordered collections of unique values

\- dict for key-value mappings



Example:



```python

name = "Alice"

age = 20

score = 95.5

is\_student = True



Lists

A list is an ordered and mutable collection. Lists can contain multiple values and can be modified after creation.



Example:

numbers = \[1, 2, 3, 4]numbers.append(5)





The append method adds an item to the end of a list.

Tuples

A tuple is an ordered and immutable collection. Once a tuple is created, its elements cannot be changed.



Example:

point = (10, 20)





Dictionaries

A dictionary stores data as key-value pairs.



Example:

student = {    "name": "Alice",    "age": 20}





Values can be accessed using their keys:

student\["name"]





Functions

A function is a reusable block of code that performs a particular task. Functions are defined using the def keyword.



Example:

def add(a, b):    return a + b





The return statement sends a value back to the caller.

Exception Handling

Python uses try and except to handle exceptions.



Example:

try:    result = 10 / 0except ZeroDivisionError:    print("Cannot divide by zero")





The except block handles the specified exception instead of allowing the program to terminate unexpectedly.

