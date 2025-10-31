from datetime import datetime

import pandas as pd

from config import Config


def load_dataset(config: Config) -> list:
    # Read the CSV file into a DataFrame, keeping the 'birth' column as strings.
    df = pd.read_csv(config.dataset_path, dtype={"birth": str})

    # Convert the DataFrame to a list of dictionaries.
    dataset = df.to_dict("records")
    dataset = post_processing(config, dataset)

    print(f"✅ {len(dataset)} records were loaded from '{config.dataset_path}'.")

    return dataset


def post_processing(config: Config, dataset: list) -> list:
    """
    Create IDs, calculate age.
    """

    person_id_map = {}
    for i, person in enumerate(dataset):
        person["id"] = i

        # Calculate age
        person["age"] = calculate_age(person["birth"])

        unique_key = f"{person['username']} - {person['birth']}"
        person_id_map[unique_key] = i

        person["disjoint_keys"] = set()

        if pd.isna(person.get("disjoint")):
            person["disjoint"] = ""

    # Set relationships for disjoint constraints.
    for person in dataset:
        if person["disjoint"]:
            disjoint_list = person["disjoint"].split("|")

            for disjoint_key in disjoint_list:
                # Remove leading/trailing whitespace from the key
                disjoint_key = disjoint_key.strip()

                # Skip if the key is now an empty string (e.g., from "||")
                if not disjoint_key:
                    continue

                if disjoint_key in person_id_map:
                    target_id = person_id_map[disjoint_key]
                    person["disjoint_keys"].add(target_id)
                else:
                    # Warn if a key doesn't match, so you can fix the data
                    print(
                        f"⚠️ WARNING: Disjoint key '{disjoint_key}' "
                        f"(for person ID {person['id']}) was not found in person_id_map. "
                        "Please check for typos in the CSV."
                    )

    df_to_save = pd.DataFrame(dataset)

    # It's cleaner to convert it to a pipe-separated string (e.g., "1|5"),
    # which matches the format of your original 'disjoint' column.
    df_to_save["disjoint_keys"] = df_to_save["disjoint_keys"].apply(
        lambda key_set: "|".join(map(str, sorted(list(key_set))))
    )

    # Reorder columns to put 'id' first
    # Get a list of all column names
    all_columns = df_to_save.columns.tolist()

    # Check if 'id' exists before trying to move it
    if "id" in all_columns:
        # Remove 'id' from its current position
        all_columns.remove("id")
        # Insert 'id' at the very beginning of the list
        new_column_order = ["id"] + all_columns
        # Apply the new column order to the DataFrame
        df_to_save = df_to_save[new_column_order]

    return dataset


def calculate_age(birth_str) -> int:
    """Calculate age from the format YYYYMMDD"""
    birth = datetime.strptime(birth_str, "%Y%m%d")
    today = datetime.today()
    return (
        today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    )


def save_solution(config: Config, dataset: list, best_solution: list) -> None:
    """
    Adds the GA solution to the dataset, sorts by Group, and saves to CSV.
    """
    config.save_as_yaml(f"./result/config_{config.run_name}.yaml")

    output_filename = f"./result/result_{config.run_name}.csv"

    # Add the solution (Group assignment) to each person
    if len(dataset) != len(best_solution):
        print(
            f"ERROR: Mismatch in lengths. Dataset has {len(dataset)} items, "
            f"but solution has {len(best_solution)} items."
        )
        return

    for i, person in enumerate(dataset):
        # We name the new column 'group'
        person["group"] = best_solution[i]

    df_final = pd.DataFrame(dataset)

    # Clean up the 'disjoint_keys' column (which is currently a set)
    if "disjoint_keys" in df_final.columns:
        df_final["disjoint_keys"] = df_final["disjoint_keys"].apply(
            lambda key_set: "|".join(map(str, sorted(list(key_set))))
        )

    # Sort the DataFrame by the 'group' column (as you requested)
    # We add 'id' as a secondary sort for a stable, clean order within groups
    df_final = df_final.sort_values(by=["group", "id"])

    # Reorder columns to put 'id' and 'group' first
    all_columns = df_final.columns.tolist()

    if "id" in all_columns:
        all_columns.remove("id")

    if "group" in all_columns:
        all_columns.remove("group")

    new_column_order = ["id", "group"] + all_columns
    df_final = df_final[new_column_order]

    try:
        df_final.to_csv(output_filename, index=False)
        print(f"✅ Successfully saved final solution to '{output_filename}'.")
    except Exception as e:
        print(f"ERROR: Failed to save solution file: {e}")
