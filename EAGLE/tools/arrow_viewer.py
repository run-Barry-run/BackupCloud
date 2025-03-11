import pyarrow as pa
import pyarrow.ipc as ipc

# Path to your .arrow file
# file_path = '.cache/huggingface/datasets/lmms-lab___you_cook2/default/0.0.0/d164962a70c383a57726388c9851e1486e9a9db7/you_cook2-val.arrow'

# csv_file_path = 'tools/youcook2_val_.csv'
# output_arrow_file = 'tools/youcook2-val.arrow'
# file_path = output_arrow_file

# file_path = '.cache/huggingface/datasets/lmms-lab___activity_net_qa/default/0.0.0/83e5a35a25748273ffcff353f161949b486925a9/activity_net_qa-test.arrow'
# csv_file_path = 'tools/activitynetqa.csv'
file_path = '.cache/huggingface/datasets/lmms-lab___you_cook2/default/0.0.0/d164962a70c383a57726388c9851e1486e9a9db7/you_cook2-test.arrow'
csv_file_path = 'tools/youcook2-test.csv'

# Read the .arrow file
with pa.memory_map(file_path, 'r') as source:
    # reader = ipc.RecordBatchFileReader(source)
    reader = ipc.RecordBatchStreamReader(source)
    
    # Get the table from the reader
    table = reader.read_all()
    
    # Print the table schema
    print("Schema:")
    print(table.schema)
    
    # Print the table contents
    print("\nTable Contents:")
    print(table.to_pandas())  # Convert to pandas DataFrame for easier viewing
    df = table.to_pandas()
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

    # # cut 
    # # Check if the table has at least 10 rows
    # if len(table) >= 1032:
    #     # Slice the table to exclude the 10th row
    #     before = table.slice(0, 1031)  # Rows 0 to 8 (first 9 rows)
    #     after = table.slice(1033)     # Rows 10 onwards
    #     # Concatenate the two parts
    #     filtered_table = pa.concat_tables([before, after])
    # else:
    #     print("The table has fewer than 10 rows. No rows were deleted.")
    #     filtered_table = table

    # # Write the modified table to a new Arrow file
    # with pa.OSFile(output_arrow_file, 'wb') as sink:
    #     writer = ipc.RecordBatchStreamWriter(sink, filtered_table.schema)
    #     writer.write_table(filtered_table)
    #     writer.close()

    # print(f"Modified Arrow file saved to: {output_arrow_file}")
