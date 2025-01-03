import pandas as pd
import json
import requests
from io import BytesIO

def ensure_hdfs_folder(hdfs_host, folder_path):
    """
    Ensure the specified folder exists in HDFS. Create it if it doesn't exist.
    """
    mkdir_url = f"{hdfs_host}/webhdfs/v1{folder_path}?op=MKDIRS"
    response = requests.put(mkdir_url)
    if response.status_code == 200:
        print(f"Folder {folder_path} exists or created successfully in HDFS.")
    else:
        print(f"Failed to ensure folder {folder_path}. Status: {response.status_code}, Response: {response.text}")

def ensure_hdfs_file(hdfs_host, folder_path, file_name):
    """
    Ensure the unique_ids file exists in HDFS.
    """
    create_url = f"{hdfs_host}/webhdfs/v1{folder_path}{file_name}?op=CREATE&overwrite=false"
    response = requests.put(create_url, allow_redirects=False)
    if response.status_code == 201:
        print(f"File {file_path} created in HDFS.")
    elif response.status_code == 307:
        redirect_url = response.headers['Location'].replace(
            "hadoop.us-central1-a.c.ccproject-419413.internal", hdfs_host
        )
        requests.put(redirect_url, data="")  # Empty file

def read_unique_ids(hdfs_host, folder_path, file_name) -> set:
    """
    Read unique IDs from the HDFS file.
    """
    read_url = f"{hdfs_host}/webhdfs/v1{folder_path}{file_name}?op=OPEN"
    response = requests.get(read_url, allow_redirects=False)
    if response.status_code == 307:
        redirect_url = response.headers['Location'].replace(
            "hadoop.us-central1-a.c.ccproject-419413.internal", hdfs_host
        )
        response = requests.get(redirect_url)
        return set(response.text.splitlines())
    return set()

def append_unique_ids(hdfs_host, folder_path, file_name, new_ids: list):
    """
    Append new unique IDs to the HDFS file.
    """
    if not new_ids:
        return
    append_url = f"{hdfs_host}/webhdfs/v1{folder_path}{file_name}?op=APPEND"
    response = requests.post(append_url, allow_redirects=False)
    if response.status_code == 307:
        redirect_url = response.headers['Location'].replace(
            "hadoop.us-central1-a.c.ccproject-419413.internal", hdfs_host
        )
        requests.post(redirect_url, data="\n".join(new_ids) + "\n")

def list_hdfs_contents(hdfs_host, folder_path):
    """
    List all contents (directories and files) in the given HDFS folder path.
    """
    list_url = f"{hdfs_host}/webhdfs/v1{folder_path}?op=LISTSTATUS"
    response = requests.get(list_url)
    if response.status_code == 200:
        content = response.json()
        files = []
        directories = []
        for item in content.get('FileStatuses', {}).get('FileStatus', []):
            if item['type'] == 'DIRECTORY':
                directories.append(item['pathSuffix'])
            elif item['type'] == 'FILE':
                files.append(item['pathSuffix'])

        print(f"Contents in {folder_path}:")
        print("Directories:")
        for directory in directories:
            print(f"  {directory}")
        print("Files:")
        for file in files:
            print(f"  {file}")
        return {"directories": directories, "files": files}
    else:
        print(f"Failed to list contents in {folder_path}. Status: {response.status_code}, Response: {response.text}")
        return {"directories": [], "files": []}

def delete_hdfs_file(hdfs_host, folder_path, file_path):
    """
    Delete a file in HDFS.
    """
    delete_url = f"{hdfs_host}/webhdfs/v1{folder_path}{file_name}?op=DELETE"
    response = requests.delete(delete_url)
    
    if response.status_code == 200:
        print(f"File {file_path} deleted successfully.")
        return True
    elif response.status_code == 404:
        print(f"File {file_path} does not exist.")
        return False
    else:
        print(f"Failed to delete file {file_path}. Status: {response.status_code}, Response: {response.text}")
        return False

def delete_hdfs_folder_content(hdfs_host, folder_path):
    """
    Delete the contents of a folder in HDFS, but keep the folder itself.
    
    Args:
    - hdfs_host (str): WebHDFS host URL.
    - folder_path (str): HDFS folder path to clear.
    """
    list_url = f"{hdfs_host}/webhdfs/v1{folder_path}?op=LISTSTATUS"
    response = requests.get(list_url)
    if response.status_code == 200:
        content = response.json()
        for item in content.get('FileStatuses', {}).get('FileStatus', []):
            item_path = f"{folder_path}/{item['pathSuffix']}"
            delete_hdfs_item(hdfs_host, item_path)
        print(f"All contents of folder {folder_path} have been deleted.")
    else:
        print(f"Failed to list contents of {folder_path}. Status: {response.status_code}, Response: {response.text}")


def delete_hdfs_item(hdfs_host, item_path):
    """
    Delete a file or folder in HDFS.
    
    Args:
    - hdfs_host (str): WebHDFS host URL.
    - item_path (str): Path to the file or folder to delete.
    """
    delete_url = f"{hdfs_host}/webhdfs/v1{item_path}?op=DELETE&recursive=true"
    response = requests.delete(delete_url)
    if response.status_code == 200:
        print(f"Deleted {item_path}")
    elif response.status_code == 404:
        print(f"{item_path} does not exist.")
    else:
        print(f"Failed to delete {item_path}. Status: {response.status_code}, Response: {response.text}")

def write_to_hdfs_json(hdfs_host, folder_path, file_name, results):
    """
    Write scraper results to HDFS as a single JSON file.
    
    Args:
    - hdfs_host (str): The WebHDFS host URL.
    - folder_path (str): The folder path in HDFS where the file will be stored.
    - file_name (str): The name of the JSON file.
    - results (dict): Scraper results as a dictionary.
    """
    # Convert results to a JSON string
    json_data = json.dumps(results, ensure_ascii=False, indent=4)
    
    # Ensure the folder exists in HDFS
    ensure_hdfs_folder(hdfs_host, folder_path)
    
    # Upload the JSON file to HDFS
    upload_to_hdfs(hdfs_host, folder_path, file_name, json_data)

def upload_to_hdfs(hdfs_host, folder_path, file_name, data):
    """
    Upload a file to HDFS using WebHDFS.
    
    Args:
    - hdfs_host (str): The WebHDFS host URL.
    - folder_path (str): The HDFS folder path.
    - file_name (str): The name of the file to upload.
    - data (str): The content to write to the file.
    """
    file_path = f"{folder_path}{file_name}"
    create_url = f"{hdfs_host}/webhdfs/v1{file_path}?op=CREATE&overwrite=true"
    response = requests.put(create_url, allow_redirects=False)
    if response.status_code == 307:
        # Follow the redirect URL
        redirect_url = response.headers['Location'].replace(
            "hadoop.us-central1-a.c.ccproject-419413.internal", hdfs_host
        )
        final_response = requests.put(redirect_url, data=data.encode('utf-8'))
        if final_response.status_code == 201:
            print(f"Successfully wrote JSON file to {file_path}")
        else:
            print(f"Failed to write JSON file to {file_path}. Status: {final_response.status_code}, Response: {final_response.text}")
    else:
        print(f"Failed to create file at {file_path}. Status: {response.status_code}, Response: {response.text}")