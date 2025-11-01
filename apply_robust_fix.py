"""
One-liner fix cho AttributeError - Safe version
"""

def apply_robust_fix():
    """Apply robust fix cho dictionary_manager.py"""
    
    file_path = '/content/Sign2VN/dictionary_manager.py'
    
    print("Reading file...")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    print("Applying fix...")
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Tìm dòng cần fix
        if "for entry in self.dictionary:" in line:
            fixed_lines.append(line)
            i += 1
            
            # Next line should be the problematic one
            if i < len(lines):
                next_line = lines[i]
                
                # Check nếu đây là dòng cũ chưa fix
                if "if entry.get('local_video'" in next_line or "local_video = entry.get('local_video')" in next_line:
                    # Replace với safe version
                    indent = len(next_line) - len(next_line.lstrip())
                    safe_code = [
                        ' ' * indent + "# Safe access to local_video\n",
                        ' ' * indent + "if not entry or not isinstance(entry, dict):\n",
                        ' ' * indent + "    continue\n",
                        ' ' * indent + "local_video = entry.get('local_video')\n",
                        ' ' * indent + "if local_video and isinstance(local_video, str) and local_video.endswith(basename):\n"
                    ]
                    
                    fixed_lines.extend(safe_code)
                    
                    # Skip original problematic lines
                    while i < len(lines) and ('local_video' in lines[i] or 'return entry' in lines[i]):
                        if 'return entry' in lines[i]:
                            fixed_lines.append(lines[i])
                            i += 1
                            break
                        i += 1
                    continue
        
        fixed_lines.append(line)
        i += 1
    
    # Write back
    print("Writing fixed file...")
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print("✓ Fix applied!")
    print("\nChanges:")
    print("- Added None check for entry")
    print("- Added dict type check")
    print("- Added string type check for local_video")
    print("- Safe endswith() call")


if __name__ == "__main__":
    print("=" * 80)
    print("ROBUST FIX FOR AttributeError")
    print("=" * 80)
    
    apply_robust_fix()
    
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    # Verify
    file_path = '/content/Sign2VN/dictionary_manager.py'
    with open(file_path, 'r') as f:
        content = f.read()
    
    if "isinstance(entry, dict)" in content and "isinstance(local_video, str)" in content:
        print("✓ Robust fix verified!")
        print("\nYou can now:")
        print("1. Restart runtime (Runtime → Restart runtime)")
        print("2. Re-run pipeline")
    else:
        print("⚠ Fix may not be complete. Please check file manually.")