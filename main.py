from A.solution import Solution as SolutionA
from B.solution import Solution as SolutionB

def print_info():
    print("-------------------------------------")
    print("|       AMLS Assignment 23-24       |")
    print("|         Name: Zhaoyan Lu          |")
    print("|        Student No: 23049710       |")
    print("-------------------------------------")
    print()

def main():
    print_info()

    print("-----------[Tasks running]-----------")
    print()

    # Task A
    solution_A = SolutionA()
    solution_A.solve()

    # Task B
    solution_B = SolutionB()
    solution_B.solve()

    print("-----------[Tasks finished]----------")


if __name__ == "__main__":
    main()
