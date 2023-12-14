import csv

def convert_txt_to_csv(txt_filename, csv_filename):
    with open(txt_filename, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split the text into lines and then into abbreviation-expansion pairs
    lines = [line.strip() for line in text.strip().split('\n')]
    pairs = [line.split('=') for line in lines]

    # Write to CSV
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Abbreviation', 'Expansion'])
        csv_writer.writerows(pairs)

    print(f"CSV file '{csv_filename}' has been created.")
