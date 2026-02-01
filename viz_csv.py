import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_curiosity(csv_path='curiosity_output.csv'):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # --- APPLY COORDINATE TRANSFORMATION ---
    # Formula: world_coord = (index * step) + offset
    m = 333.33
    b = -8002.0
    
    df['x_world'] = (df['x'] * m + b).round().astype(int)
    df['z_world'] = (df['z'] * m + b).round().astype(int)
    # ---------------------------------------

    grouped = df.groupby(['level'])
    # grouped = df.groupby(['course', 'act', 'area'])


    if not os.path.exists('plots'):
        os.makedirs('plots')

    for (level), data in grouped:
        # print(f"Processing Course {course}, Act {act}, Area {area}...")
        print(f"Processing level {level}")

        # Collapse the Y dimension (Verticality) using world coordinates
        top_down = data.groupby(['x_world', 'z_world'])['visits'].sum().reset_index()

        # Create Pivot Table
        grid = top_down.pivot(index='z_world', columns='x_world', values='visits')

        plt.figure(figsize=(12, 10))
        
        # Heatmap
        sns.heatmap(
            grid, 
            cmap='magma', 
            cbar_kws={'label': 'Total Visits'},
            # Use fewer labels to prevent overlap with floating point numbers
            xticklabels=10, 
            yticklabels=10
        )

        # plt.title(f"SM64 Heatmap: Course {course}, Act {act}, Area {area}\n(World Coordinates)")
        plt.title(f"SM64 Heatmap: level {level}\n(World Coordinates)")

        plt.xlabel("X Position (SM64 Units)")
        plt.ylabel("Z Position (SM64 Units)")
        
        plt.gca().invert_yaxis()

        # filename = f"plots/map_world_C{course}_A{act}_Area{area}.png"
        filename = f"plots/map_world_L{level}.png"

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    visualize_curiosity()