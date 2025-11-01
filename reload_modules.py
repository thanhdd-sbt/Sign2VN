"""
Force reload modules và verify fix
"""

import sys
import importlib


def reload_all_modules():
    """Force reload tất cả custom modules"""
    print("Reloading all modules...")
    
    modules_to_reload = [
        'labeling_config',
        'dictionary_manager',
        'video_scanner', 
        'landmark_extractor',
        'data_labeling_pipeline'
    ]
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            print(f"  Reloading {module_name}...")
            importlib.reload(sys.modules[module_name])
        else:
            print(f"  {module_name} not loaded yet")
    
    print("✓ Reload complete!")


def verify_fix():
    """Verify rằng fix đã được apply"""
    print("\nVerifying fix...")
    
    # Read file
    file_path = '/content/Sign2VN/dictionary_manager.py'
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for new code
    if "local_video = entry.get('local_video') or ''" in content:
        print("✓ Fix đã được apply vào file!")
        return True
    else:
        print("✗ Fix chưa được apply!")
        print("\nApplying fix now...")
        
        # Apply fix
        old_code = "if entry.get('local_video', '').endswith(basename):"
        new_code = """local_video = entry.get('local_video') or ''
            if local_video and local_video.endswith(basename):"""
        
        if old_code in content:
            content = content.replace(old_code, new_code)
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            print("✓ Fix applied!")
            return True
        else:
            print("✗ Could not find code to fix. Manual intervention needed.")
            return False


if __name__ == "__main__":
    print("=" * 80)
    print("MODULE RELOAD & FIX VERIFICATION")
    print("=" * 80)
    
    # Verify fix
    if verify_fix():
        # Reload modules
        reload_all_modules()
        
        print("\n" + "=" * 80)
        print("✓ READY TO RUN!")
        print("=" * 80)
        print("\nNow you can run:")
        print("  from data_labeling_pipeline import DataLabelingPipeline")
        print("  pipeline = DataLabelingPipeline()")
        print("  pipeline.run(resume=True)")
    else:
        print("\n" + "=" * 80)
        print("✗ FIX FAILED")
        print("=" * 80)
        print("\nPlease:")
        print("1. Download dictionary_manager.py from outputs folder")
        print("2. Upload to /content/Sign2VN/")
        print("3. Restart runtime")