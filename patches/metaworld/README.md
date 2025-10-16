# MetaWorld Compatibility Patch

## Apply Patch

```bash
bash patches/metaworld/apply_patches.sh
```

## What It Does

Patches `gymnasium/envs/mujoco/mujoco_env.py` to enable MuJoCo 2.x/3.x coexistence by commenting out strict `render_modes` assertion. Required for MetaWorld to work with LeRobot.

## Restore Original

```bash
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cp "$SITE_PACKAGES/gymnasium/envs/mujoco/mujoco_env.py.bak" \
   "$SITE_PACKAGES/gymnasium/envs/mujoco/mujoco_env.py"
```
