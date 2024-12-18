# CSV Loading and processing

This process extracts and filters patient and examination data to identify which patients played the MAX_FORCE game and which fingers were tested. The final output links patient codes to `id` from `step_result` and ensures only those who responded to a questionnaire are included.

<b>Workflow:</b>

1. **Filter Game Data**:
   - From `step.csv`, extract the `id` and `fingers` columns only for rows where the game type is `MAX_FORCE`.
    
2. **Match Patient Codes**:
   - Extract patient codes in the format `XXXX-XXXX` from `RehapianoResponses.xlsx`.
   - Match these codes with the `id` column in `patient.csv` using the `code` .

3. **Identify Valid Examinations**:
   - Use `examination_id` to store patient IDs that:
     - Match the previously identified patient codes.
     - Indicate that the patient underwent an examination and responded to the questionnaire.

4. **Link Examination Data**:
   - Match the `examination_id` from the previous step with the `examination_id` in `step_results.csv`.
   - Retrieve the primary IDs from the `id` column in `step_results.csv`.

5. **Final Output**:
   - Return `final_ids`, which includes:
     - Primary IDs.
     - Corresponding finger data for each ID.

<b>Notes:</b>
- **Primary Key**: The `id` column in `step_results.csv` serves as the main identifier.
- **Critical Filtering**: Ensures that only patients who:
  - Played the MAX_FORCE game.
  - Responded to the questionnaire.
  - Underwent an examination.
  Are included in the final output.


# Finger Pressure Data Visualization
 <b>Inputs:</b>

- **`data_block`**: (`[[float]]`), where:
  - Each sublist represents the pressure values applied by one finger over time.
  - Contains 10 sublists, one for each finger.
- **`fingers`**: (`[str(int)]`), representing specific fingers, used to correctly assign data from `data_block` to specific fingers.

 <b>Output:</b>

 Scatter plots for each sublist in `data_block`, labeled by their corresponding ID.

 <b>Finger mapping:</b>

The finger indices in `fingers` correspond to the following names:
`0`: Pinky (Left Hand), `1`: Ring (Left Hand) .... `9`: Pinky (Right Hand)

 <b>Notes:</b>
- **Hand Type**: Determines which hand(s) the data represents:
  - **Left Hand**: Indices `0-4`.
  - **Right Hand**: Indices `5-9`.


# Calculations and statistics for MAX finger and hand force

These functions process and analyze finger force data, calculating cumulative forces and smoothed metrics for further analysis.

<b>`calculate_raw_max_values(group)`:</b>
- **Purpose**: Calculates the cumulative force applied at each time point across multiple arrays, representing either the left or right hand.
- **Inputs**:
  - **`group`**: A list of 5 arrays[[float]], each representing the force data for one finger of either the left or right hand.
- **Output**:
  - [float] each value represents the total force applied across all fingers at that specific time point.

<b>`calculate_metrics(data_block, fingers, window_size=5)`:</b>
- **Purpose**: Calculates smoothed maximum force values and moving averages for each finger.
- **Inputs**:
  - **`data_block`**: `[[float]]`, where each array contains force values over time for a specific finger.
  - **`fingers`**: `[str(int)]` for the fingers corresponding to the arrays in `data_block`.
  - **`window_size`** (optional): The size of the window used for moving average smoothing. Default is 5.
- **Outputs**:
  - **`max_values`**: A `{str : float}` where keys are finger identifiers and values are the maximum smoothed values.
  - **`moving_averages`**: A `{str : [float]}` where keys are finger identifiers and values are arrays of smoothed values using the moving average.


# .BIN file processing
## Functions for Decompressing, Processing, and Analyzing Finger Force Data

<b>`decompress_file(file_path)`:</b>
Decompress and deserialize a `.BIN` file containing raw server data.
- **Inputs**:
  - **`file_path`**: Path to the `.BIN` file to decompress.
- **Output**:
  - A deserialized Python object, in list, or `None` if decompression or deserialization fails.
---

<b>`get_tensometer_indices(fingers)`:</b>
Determine the tensometer indices to process based on the fingers used.
- **Inputs**:
  - **`fingers`**: `[str(int)]`representing finger indices.
- **Output**:
  - A list of tensometer indices corresponding to the provided fingers:
    - **Left hand**: `[0, 1, 2, 3, 4]`
    - **Right hand**: `[12, 11, 10, 9, 8]`
    - Empty list if the configuration is invalid.

---

<b>`process_server_data(server_data, tensometer_indices)`:</b> 
 Process raw server data and extract aggregated tensometer values.
- **Inputs**:
  - **`server_data`**: `[['bytes']]` List of attempts from the decompressed file.
  - **`tensometer_indices`**: `[int]` representing fingers as mentioned above
- **Output**:
  - `step_agg`:`[[[float]]]` A list of aggregated tensometer data arrays.
- **Notes**:
  - Handles decoding using `reha_conv.decode`.

## Functions for Decompressing, Processing, and Analyzing Finger Force Data

<b>`decompress_file(file_path)`:</b>
Decompress and deserialize a `.BIN` file containing raw server data.
- **Inputs**:
  - **`file_path`**: Path to the `.BIN` file to decompress.
- **Output**:
  - A deserialized Python object, in list, or `None` if decompression or deserialization fails.
---

<b>`get_tensometer_indices(fingers)`:</b>
Determine the tensometer indices to process based on the fingers used.
- **Inputs**:
  - **`fingers`**: `[str(int)]`representing finger indices.
- **Output**:
  - A list of tensometer indices corresponding to the provided fingers:
    - **Left hand**: `[0, 1, 2, 3, 4]`
    - **Right hand**: `[12, 11, 10, 9, 8]`
    - Empty list if the configuration is invalid.

---

<b>`process_server_data(server_data, tensometer_indices)`:</b> 
 Process raw server data and extract aggregated tensometer values.
- **Inputs**:
  - **`server_data`**: `[['bytes']]` List of attempts from the decompressed file.
  - **`tensometer_indices`**: `[int]` representing fingers as mentioned above
- **Output**:
  - `step_agg`:`[[[float]]]` A list of aggregated tensometer data arrays.
- **Notes**:
  - Handles decoding using `reha_conv.decode`.


# Write max force data to CSV

<b>`write_results_to_csv(results, csv_file_path, finger_count=10)`:</b>  

This function writes the processed results into a CSV file, dynamically including data for each finger and overall hand metrics.

<b>Data Written to CSV: </b>
1. **`filename`**:
   - The name of the processed `.BIN` file, which is the primary key, corresponding to `id` in `step_result.csv`

2. **`max_force_left_hand`**:
   - Calculated using `calculate_raw_max_values`. The smoothed maximum force value for the left hand.

3. **`max_force_right_hand`**:
   -  Calculated using `calculate_raw_max_values`. The smoothed maximum force value for the right hand.

4. **`Finger {i}` (for each finger) `hand_moving_average`**:
   - Source: Values from `max_values_by_finger` in the `results` dictionary.The smoothed maximum force value for each individual finger.

<b>Notes: </b>
- The fieldnames are dynamically generated to include columns for all fingers (`Finger 0`, `Finger 1`, ..., `Finger {finger_count - 1}`).
- Missing or invalid data is handled as `"nan"` in the output.


# Calculate reaction offset

<b>`filter_rows_by_additional_data(input_file, output_file)`</b>

Filters rows from a CSV file where the `additional_data` column contains a non-empty list and writes the filtered rows to `reaction_offset.csv`, that will be used by function `process_reaction_offset()`.

<b>Workflow: </b>
   - Evaluates the content of the `additional_data` column.
   - Checks if the column contains a non-empty list.
   - Writes rows with valid `additional_data` to the `reaction_offset.csv`.

---

<b> `validate_additional_data(additional_data)`</b>

Validates and parses the `additional_data` field to ensure it matches the expected structure.

<b>Inputs: </b>
- data from the `additional_data` column.

<b>Output: </b>
- `[[{str: float}]]` data from the `additional_data` column.
- `None` if the data does not match the expected structure.

<b>Validation Criteria: </b>
1. The `additional_data` must evaluate to a list.
2. Each element in the list must:
   - Be a list itself.
   - Contain dictionaries with the keys `"finger"` and `"reaction"`.
---

<b> `calculate_reaction_offset_metrics(additional_data_list)`</b>

This function processes reaction data for each finger to calculate:
1. Mean reaction times.
2. Reaction percentages.
3. Average reaction times for the left and right hands.

<b>Inputs: </b>
- **`additional_data_list`**: `[[{str: float}]]`:
- **`finger`**: An integer (0–9) indicating the finger.
- **`reaction`**: A numerical value representing the reaction time (0 if no reaction occurred).

<b>Output: </b>
A dictionary containing:
- **Mean Reaction Times** (per finger): Mean of all non-zero reaction times for the finger.

- **Reaction Percentages** (per finger):Percentage of total stimuli that resulted in a non-zero reaction.

- **Left, Right Hand Average**: Mean reaction time across all fingers on the left/right hand

# Write reaction offset data to CSV


<b>`process_reaction_offset(input_file, output_file)`</b>

This function processes reaction offset data from  `reaction_offset.csv`, calculates averages and percentages for each finger, and saves the results to `reaction_offset_result.csv`, including left and right hand averages.

<b>Inputs: </b>
filepaths

<b>Outputs: </b>
 A CSV file containing the following fields for each row:
  - **`ID_filename`**: The `id` from the `reaction_offset.csv`, which is primary key.
  - **`Finger {i}`**: Mean reaction times for each finger (0–9).
  - **`Finger {i} (%)`**: Reaction percentages for each finger (0–9).
  - **`Left Hand Average`**: Average reaction time for fingers 0–4.
  - **`Right Hand Average`**: Average reaction time for fingers 5–9.

<b>Workflow: </b>
- Validate data from `additional_data` column using function : `validate_additional_data`
- Calculate Metrics(mean, reaction percentages) using function: `calculate_reaction_offset_metrics`
- Save Results to `reaction_offset-result.csv`

# Main 
Runs above blocks as follows:
-  CSV Loading and processing
-  .BIN file processing -> Calculations and statisctics for MAX finger and hand force, Finger Pressure Data Visualization
-  Write max force data to CSV
-  Calculate reaction offset
-  Write reaction offset data to CSV



# `dataloader(input_folder, output_folder)`

This function processes JSON files from the specified input folder, performs transformations on the content (such as decoding Base64 data), and saves the processed files to the output folder.

## Inputs
- **`input_folder`**: The path to the folder containing input JSON files.
- **`output_folder`**: The path to the folder where processed JSON files will be saved.

## Outputs
A set of JSON files saved in the `output_folder` with the following transformations applied:
- Any `"raw"` field (if present) is decoded from Base64 and stored as `"decoded_raw"`.
- Files that are not valid JSON or do not contain a `"raw"` field are skipped.

## Workflow
1. **Ensure Output Folder Exists**:
   - Automatically creates the `output_folder` if it does not already exist.

2. **Process Each File**:
   - Reads each file from the `input_folder`.
   - Skips files that are not valid JSON.

3. **Decode Base64 Data**:
   - If a file contains a `"raw"` key:
     - Decodes the Base64 content of `"raw"` into UTF-8.
     - Stores the decoded value in a new key, `"decoded_raw"`.
     - Logs any errors during the decoding process and includes the error message in the output.

4. **Save Results**:
   - Saves the processed JSON to the `output_folder` with the same filename as the input file.
   
   
   
# Project Contributions

## Matej Labaj
- **Facilitated the Rehapiano examination**: Collected relevant data and managed the process.
- **Developed visualization code**: Created code to visualize the maximum force exerted by fingers, providing a clear representation of the data.
- **Identified and mapped fingers**: Accurately identified and mapped individual fingers in the data set.
- **Integrated data with subject codes**: Linked the subjects' code with their corresponding data.
- **Wrote documentation**: Documented the visualization and integration processes.
- **Calculated maximum compression force**:  
  - For the left and right hands, determined as the average of a 5-second measurement.  
  - For individual fingers, determined as the average of a 5-second measurement.

## Lucia Helmeciova
- **Communicated with stakeholders**: Acted as the main point of contact with Jan Magyar, gathering detailed information on data specifications and requirements.
- **Wrote reaction offset code**: Implemented calculations for:  
  - Average reaction speed for each finger.  
  - Reaction success rate for each finger.
- **Refactored codebase**: Improved code readability and maintainability.
- **Integrated data with subject codes**: Worked on connecting the subjects' code with their corresponding data.
- **Wrote documentation**: Documented reaction offset calculations and refactoring process.

## Tomáš Gamrat
- **Processed examination data**: Wrote code to process data from the examination.
- **Designed candlestick diagrams**: Created visual representations of data trends and anomalies.
- **Performed data processing tasks**:  
  - Processed binary data.  
  - Loaded and processed JSON data.  
  - Established connections between subject codes and their associated data.
- **Communicated with stakeholders**: Worked closely with Maroš Hliboký and Jan Magyar on data handling and project requirements.
- **Calculated maximum compression force**:  
  - For the left and right hands, determined as the average of a 5-second measurement.  
  - For individual fingers, determined as the average of a 5-second measurement.

