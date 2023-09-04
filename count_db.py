import sqlite3

def count_rows(database_path, table_name):
    conn = sqlite3.connect(database_path)
    c = conn.cursor()

    # Execute the counting query
    c.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = c.fetchone()[0]

    conn.close()
    return count

if __name__ == "__main__":
    database_path = "augmented.db"  # replace with the path to your SQLite database
    table_name = "augmented"  # replace with your table name
    print(f"Number of rows in {table_name}: {count_rows(database_path, table_name)}")

