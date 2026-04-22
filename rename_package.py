
import os


def replace_in_files(root_dir, old_str, new_str):
    for dirpath, _, filenames in os.walk(root_dir):
        if ".git" in dirpath or "dist" in dirpath or ".egg-info" in dirpath or "__pycache__" in dirpath:
            continue
            
        for filename in filenames:
            if not filename.endswith((".py", ".md", ".toml", ".txt")):
                continue
                
            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, encoding="utf-8") as f:
                    content = f.read()
                
                if old_str in content:
                    print(f"Updating {filepath}")
                    new_content = content.replace(old_str, new_str)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(new_content)
            except Exception as e:
                print(f"Skipping {filepath}: {e}")

if __name__ == "__main__":
    replace_in_files(".", "valencity", "valencity")
    replace_in_files(".", "valencity", "valencity")
