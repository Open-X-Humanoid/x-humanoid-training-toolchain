#!/bin/bash
# Apply MetaWorld compatibility patch for gymnasium
# Run from project root: bash patches/metaworld/apply_patches.sh

set -e

echo "🔧 Applying MetaWorld compatibility patch..."

# Detect Python site-packages directory
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
GYMNASIUM_MUJOCO_ENV="$SITE_PACKAGES/gymnasium/envs/mujoco/mujoco_env.py"

echo "Site-packages: $SITE_PACKAGES"

# Check if gymnasium is installed
if [ ! -f "$GYMNASIUM_MUJOCO_ENV" ]; then
    echo "❌ Error: gymnasium not found. Please install it first:"
    echo "   pip install gymnasium"
    exit 1
fi

# Backup original file if not already backed up
if [ ! -f "${GYMNASIUM_MUJOCO_ENV}.bak" ]; then
    echo "📦 Creating backup: ${GYMNASIUM_MUJOCO_ENV}.bak"
    cp "$GYMNASIUM_MUJOCO_ENV" "${GYMNASIUM_MUJOCO_ENV}.bak"
else
    echo "ℹ️  Backup already exists, skipping..."
fi

# Apply patch
PATCH_FILE="$(dirname "$0")/gymnasium_mujoco_env.patch"

if [ ! -f "$PATCH_FILE" ]; then
    echo "❌ Error: Patch file not found at $PATCH_FILE"
    exit 1
fi

echo "🔨 Applying patch to gymnasium..."

# Try to apply patch
if patch -p1 -d "$SITE_PACKAGES" --forward --dry-run < "$PATCH_FILE" > /dev/null 2>&1; then
    patch -p1 -d "$SITE_PACKAGES" --forward < "$PATCH_FILE"
    echo "✅ Patch applied successfully!"
elif grep -q "# Custom modification: Commented out render_modes assertion" "$GYMNASIUM_MUJOCO_ENV" 2>/dev/null; then
    echo "ℹ️  Patch already applied, skipping..."
else
    echo "⚠️  Patch may already be applied or file has been modified."
    echo "    Manual verification recommended."
fi

echo ""
echo "🎉 Compatibility patches applied successfully!"
echo ""
echo "Why this patch is needed:"
echo "  MetaWorld requires MuJoCo 2.x compatibility but LeRobot uses MuJoCo 3.x."
echo "  This patch removes the strict render_modes assertion in gymnasium to"
echo "  allow MetaWorld environments to work with the newer MuJoCo version."
echo ""
