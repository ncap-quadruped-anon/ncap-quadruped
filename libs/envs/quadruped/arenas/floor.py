from dm_control.locomotion import arenas


class Floor(arenas.Floor):
  def _build(self, aesthetic='default', **kwargs):
    super()._build(aesthetic=aesthetic, **kwargs)

    if aesthetic == 'default':
      self._skybox = self._mjcf_root.asset.add(
        'texture',
        name='skybox',
        type='skybox',
        builtin='gradient',
        rgb1='0.3 0.5 0.7',
        rgb2='0 0 0',
        width='512',
        height='3072',
      )

    self._top_camera.remove()
    del self._top_camera

    self._light_ceiling = self._mjcf_root.worldbody.add(
      'light',
      name="ceiling",
      pos="0 0 3",
      dir="0 0 -1",
      directional="true",
    )
