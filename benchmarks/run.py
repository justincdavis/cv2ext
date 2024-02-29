from code import test_both, test_read, test_show, test_naive


if __name__ == "__main__":
    results = {
        "read": [],
        "naive": [],
        "show": [],
        "both": [],
    }
    for _ in range(2):
        results["read"].append(test_read())
        results["naive"].append(test_naive())
        results["show"].append(test_show())
        results["both"].append(test_both())
    
    print("Naive:", sum(results["naive"]) / len(results["naive"]))
    print("Read:", sum(results["read"]) / len(results["read"]))
    print("Show:", sum(results["show"]) / len(results["show"]))
    print("Both:", sum(results["both"]) / len(results["both"]))
