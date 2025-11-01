"""
Script để tìm files trong Google Drive (including Shared with me)
"""

import os
from pathlib import Path


def find_file_in_drive(filename, search_root="/content/drive"):
    """
    Tìm file trong toàn bộ Google Drive (bao gồm Shared with me)
    
    Args:
        filename: Tên file cần tìm
        search_root: Root directory để search
    
    Returns:
        List of matching paths
    """
    print(f"Searching for '{filename}' in {search_root}...")
    print("This may take a few minutes...")
    
    matches = []
    
    for root, dirs, files in os.walk(search_root):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        if filename in files:
            full_path = os.path.join(root, filename)
            matches.append(full_path)
            print(f"  ✓ Found: {full_path}")
    
    return matches


def find_folder_in_drive(foldername, search_root="/content/drive"):
    """
    Tìm folder trong Google Drive
    
    Args:
        foldername: Tên folder cần tìm
        search_root: Root directory để search
    
    Returns:
        List of matching paths
    """
    print(f"Searching for folder '{foldername}' in {search_root}...")
    
    matches = []
    
    for root, dirs, files in os.walk(search_root):
        if foldername in dirs:
            full_path = os.path.join(root, foldername)
            matches.append(full_path)
            print(f"  ✓ Found: {full_path}")
    
    return matches


def check_shared_folders():
    """
    Check các shared folders có sẵn
    """
    print("\n" + "=" * 80)
    print("CHECKING GOOGLE DRIVE STRUCTURE")
    print("=" * 80)
    
    drive_root = "/content/drive"
    
    if not os.path.exists(drive_root):
        print("❌ Drive not mounted! Please run:")
        print("   from google.colab import drive")
        print("   drive.mount('/content/drive')")
        return None
    
    print(f"\n✓ Drive mounted at: {drive_root}\n")
    
    # Check common paths
    paths_to_check = {
        "My Drive": f"{drive_root}/MyDrive",
        "Shared drives": f"{drive_root}/Shareddrives",
        ".shortcut-targets-by-id": f"{drive_root}/.shortcut-targets-by-id"
    }
    
    for name, path in paths_to_check.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"{exists} {name}: {path}")
        
        if os.path.exists(path):
            try:
                items = os.listdir(path)
                print(f"   Contains {len(items)} items")
                if len(items) <= 10:
                    for item in items[:5]:
                        print(f"     - {item}")
            except:
                pass
    
    return drive_root


def find_thanhnv_data():
    """
    Tìm folder users/thanhnv/data
    """
    print("\n" + "=" * 80)
    print("SEARCHING FOR 'users/thanhnv/data'")
    print("=" * 80)
    
    # Search patterns
    patterns = [
        "thanhnv",
        "data",
        "videos",
        "dictionary.json"
    ]
    
    results = {}
    
    for pattern in patterns:
        print(f"\nSearching for: {pattern}")
        
        if pattern.endswith('.json'):
            matches = find_file_in_drive(pattern)
        else:
            matches = find_folder_in_drive(pattern)
        
        results[pattern] = matches
        
        if not matches:
            print(f"  ✗ Not found")
    
    return results


def generate_correct_paths(search_results):
    """
    Tạo đúng paths dựa trên search results
    """
    print("\n" + "=" * 80)
    print("SUGGESTED PATHS")
    print("=" * 80)
    
    # Tìm dictionary.json path
    dict_paths = search_results.get('dictionary.json', [])
    
    if dict_paths:
        dict_path = dict_paths[0]
        print(f"\n✓ Found dictionary.json at:")
        print(f"  {dict_path}")
        
        # Extract relative path từ /content/drive/
        rel_path = dict_path.replace('/content/drive/', '')
        print(f"\n📝 Update DICTIONARY_PATH in labeling_config.py to:")
        print(f'  DICTIONARY_PATH = "{rel_path}"')
        
        # Tìm videos folder
        data_dir = os.path.dirname(dict_path)
        videos_dir = os.path.join(data_dir, "videos")
        videos_nnkh_dir = os.path.join(data_dir, "videos_nnkh")
        
        print(f"\n📝 Update SHARED_FOLDERS in labeling_config.py to:")
        
        folders = []
        if os.path.exists(videos_dir):
            rel_videos = videos_dir.replace('/content/drive/', '')
            print(f'  "{rel_videos}",')
            folders.append(rel_videos)
        
        if os.path.exists(videos_nnkh_dir):
            rel_videos_nnkh = videos_nnkh_dir.replace('/content/drive/', '')
            print(f'  "{rel_videos_nnkh}",')
            folders.append(rel_videos_nnkh)
        
        return {
            'dictionary_path': rel_path,
            'shared_folders': folders
        }
    
    return None


def create_shortcuts_guide():
    """
    Hướng dẫn tạo shortcuts
    """
    print("\n" + "=" * 80)
    print("ALTERNATIVE: CREATE SHORTCUTS IN MY DRIVE")
    print("=" * 80)
    
    print("""
Nếu bạn muốn dùng paths đơn giản hơn, hãy tạo shortcut:

1. Mở Google Drive trên web browser
2. Vào "Shared with me"
3. Tìm folder "users" (hoặc folder chứa data)
4. Right-click → "Add shortcut to Drive"
5. Chọn "My Drive" → "Add shortcut"

Sau đó paths sẽ là:
  MyDrive/users/thanhnv/data/dictionary.json
  MyDrive/users/thanhnv/data/videos
  MyDrive/users/thanhnv/data/videos_nnkh

Và config sẽ hoạt động như ban đầu!
    """)


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("GOOGLE DRIVE PATH FINDER")
    print("=" * 80)
    
    # Check drive
    drive_root = check_shared_folders()
    
    if not drive_root:
        return
    
    # Search for data
    results = find_thanhnv_data()
    
    # Generate correct paths
    correct_paths = generate_correct_paths(results)
    
    if correct_paths:
        print("\n" + "=" * 80)
        print("✓ SOLUTION FOUND!")
        print("=" * 80)
        
        print(f"\nDictionary: {correct_paths['dictionary_path']}")
        print(f"Shared folders:")
        for folder in correct_paths['shared_folders']:
            print(f"  - {folder}")
        
        # Auto-update config
        print("\n" + "=" * 80)
        print("AUTO-UPDATE CONFIG")
        print("=" * 80)
        
        config_update = f"""
# Copy đoạn này vào labeling_config.py:

# Shared folders
SHARED_FOLDERS = [
"""
        for folder in correct_paths['shared_folders']:
            config_update += f'    "{folder}",\n'
        
        config_update += f"""]

# Dictionary path
DICTIONARY_PATH = "{correct_paths['dictionary_path']}"
"""
        
        print(config_update)
        
        # Save to file
        with open('/tmp/labeling_config_update.txt', 'w') as f:
            f.write(config_update)
        
        print("\n✓ Saved config update to: /tmp/labeling_config_update.txt")
    else:
        print("\n" + "=" * 80)
        print("❌ COULD NOT FIND FILES AUTOMATICALLY")
        print("=" * 80)
        
        create_shortcuts_guide()
        
        print("\nOr manually search for files:")
        print("  !find /content/drive -name 'dictionary.json' 2>/dev/null")


if __name__ == "__main__":
    main()
