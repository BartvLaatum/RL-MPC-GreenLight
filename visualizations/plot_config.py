import matplotlib.pyplot as plt

# Ensure text is converted to paths
plt.rcParams["svg.fonttype"] = "path"  # Converts text to paths
plt.rcParams["text.usetex"] = False    # Use mathtext instead of full LaTeX

# Sans-serif fonts for Adobe Illustrator (labels, ticks, legend)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"]
# plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["mathtext.fontset"] = "stix"  # STIX math fonts (Times-compatible)

plt.rcParams["font.size"] = 8 # General font size
plt.rcParams["axes.labelsize"] = 8 # Axis label size
plt.rcParams["xtick.labelsize"] = 8 # Tick labels
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8

plt.rcParams["text.usetex"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Alternative: Use full LaTeX with Times-compatible fonts (requires LaTeX installation)
# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

plt.rcParams["axes.linewidth"] = 1.5  # Axis border thickness
plt.rcParams["lines.linewidth"] = 1.5  # Line thickness
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.width"] = 1
plt.rcParams["axes.grid"] = False

# Hatch settings for Adobe Illustrator SVG compatibility
plt.rcParams["hatch.linewidth"] = 1.0  # Thicker hatch lines
plt.rcParams["hatch.color"] = "white"  # Explicit hatch color
