import sqlite3

def add_row_and_delete_rows():
    # Connect to the database
    conn = sqlite3.connect('gesture_db.db')
    c = conn.cursor()

    # Add a new row with g_id 0 and g_name "You look great"
    c.execute("INSERT INTO gesture (g_id, g_name) VALUES (?, ?);", (0, "You look great"))

    # Delete rows with g_id from 6 to 10
    c.execute("DELETE FROM gesture WHERE g_id BETWEEN 6 AND 10;")

    # Commit the changes
    conn.commit()

    # Close the connection
    conn.close()

# Call the function to add a new row and delete rows with g_id from 6 to 10
add_row_and_delete_rows()
