import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os

# 設定路徑
pic_dir = os.path.join(os.getcwd(), 'day29', 'pic')
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

def draw_streamlit_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)

    # --- 1. User / Developer (Left) ---
    ax.text(2, 7, "1. Developer", fontsize=14, fontweight='bold', ha='center')
    
    # Python Script File
    rect_script = FancyBboxPatch((1, 4.5), 2, 2, boxstyle="round,pad=0.1", fc="#FFE082", ec="black") # Yellow
    ax.add_patch(rect_script)
    ax.text(2, 5.5, "app.py\n(Python Code)", ha='center', va='center', fontsize=12)
    
    # Code content example
    ax.text(2, 5.0, "st.title()\nst.image()", ha='center', va='center', fontsize=8, style='italic', color='gray')

    # --- 2. Streamlit Backend (Middle) ---
    ax.text(7, 7, "2. Streamlit Server (Backend)", fontsize=14, fontweight='bold', ha='center')
    
    # Server Box
    rect_server = FancyBboxPatch((5, 3), 4, 3.5, boxstyle="round,pad=0.2", fc="#FF5252", ec="black") # Red
    ax.add_patch(rect_server)
    ax.text(7, 6.0, "Python Interpreter", ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Logic inside server
    ax.text(7, 5.0, "Executes Script\nTop to Bottom", ha='center', va='center', fontsize=10, color='white')
    ax.text(7, 4.0, "Translates to\nFrontend Events", ha='center', va='center', fontsize=10, color='white')

    # --- 3. Web Browser (Right) ---
    ax.text(12, 7, "3. Web Browser (Frontend)", fontsize=14, fontweight='bold', ha='center')
    
    # Browser Window
    rect_browser = FancyBboxPatch((10, 3), 4, 3.5, boxstyle="round,pad=0.1", fc="#448AFF", ec="black") # Blue
    ax.add_patch(rect_browser)
    ax.text(12, 6.0, "User Interface", ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # UI Elements
    rect_btn = patches.Rectangle((11, 4.8), 2, 0.5, fc="white", ec="black")
    ax.add_patch(rect_btn)
    ax.text(12, 5.05, "Button", ha='center', va='center', fontsize=8)
    
    rect_img = patches.Rectangle((11, 3.5), 2, 1, fc="white", ec="black")
    ax.add_patch(rect_img)
    ax.text(12, 4.0, "Image", ha='center', va='center', fontsize=8)

    # --- Arrows & Data Flow ---
    
    # 1. Run Script
    ax.annotate("", xy=(5, 5.5), xytext=(3, 5.5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(4, 5.7, "Run", ha='center', fontsize=10)

    # 2. Send UI Structure (DeltaGenerator)
    ax.annotate("", xy=(10, 5.5), xytext=(9, 5.5), arrowprops=dict(arrowstyle="->", lw=2, color='green'))
    ax.text(9.5, 5.7, "JSON / Proto", ha='center', fontsize=10, color='green')

    # 3. User Interaction (Trigger Re-run)
    ax.annotate("", xy=(9, 3.5), xytext=(10, 3.5), arrowprops=dict(arrowstyle="->", lw=2, color='orange', linestyle='--'))
    ax.text(9.5, 3.7, "Event (Click)", ha='center', fontsize=10, color='orange')
    
    # Re-run loop
    ax.annotate("", xy=(7, 3), xytext=(7, 2.5), arrowprops=dict(arrowstyle="-", lw=2, color='orange', linestyle='--'))
    ax.annotate("", xy=(2, 2.5), xytext=(7, 2.5), arrowprops=dict(arrowstyle="-", lw=2, color='orange', linestyle='--'))
    ax.annotate("", xy=(2, 4.5), xytext=(2, 2.5), arrowprops=dict(arrowstyle="->", lw=2, color='orange', linestyle='--'))
    ax.text(4.5, 2.7, "Trigger Re-run", ha='center', fontsize=10, color='orange', fontweight='bold')

    # --- Explanation Text ---
    plt.figtext(0.5, 0.15, "The Streamlit Cycle: Interaction -> Re-run Script -> Update UI", ha="center", fontsize=14, fontweight='bold')
    plt.figtext(0.5, 0.1, "No HTML/JS needed. Streamlit handles the translation.", ha="center", fontsize=12)

    plt.savefig(os.path.join(pic_dir, '29-2_Streamlit_Architecture.png'))
    print("Streamlit Architecture plot saved.")

if __name__ == "__main__":
    draw_streamlit_architecture()
