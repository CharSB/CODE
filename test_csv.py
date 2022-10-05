import csv


def main():
    f = open("test.csv", "w")
    with open('test.csv', 'w', newline='') as csvfile:
        fieldnames = ['s1', 's2', 'rscore']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'s1': 56, 's2': 34, 'rscore': 0.987})
  

if __name__ == "__main__":
    main()