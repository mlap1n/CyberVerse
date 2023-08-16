from dm_control import composer
from dm_control import mjcf


class BaseArena(composer.Arena):
    """Suite-specific subclass of the base Composer arena."""
    def _build(self,
               name=None,
               size=(8, 8),
               reflectance=.2,
               groundplane_quad_size=0.25
               ):
        """Initializes this arena.
        name: (optional) A string, the name of this arena. If `None`, use the
            model name defined in the MJCF file.
        """
        super()._build(name=name)
        self._size = size

        self._mjcf_root.visual.headlight.set_attributes(
            ambient=[.4, .4, .4], diffuse=[.8, .8, .8], specular=[.1, .1, .1])


        # Add visual assets.
        # add sky
        self.mjcf_model.asset.add(
            'texture',
            type='skybox',
            builtin='gradient',
            rgb1=(0.4, 0.6, 0.8),
            rgb2=(0., 0., 0.),
            width=100,
            height=100)

        self._groundplane_texture = self.mjcf_model.asset.add(
            'texture',
            name='groundplane',
            type='2d',
            builtin='checker',
            rgb1=(0.2, 0.3, 0.4),
            rgb2=(0.1, 0.2, 0.3),
            width=200,
            height=200,
            mark='edge',
            markrgb=(0.8, 0.8, 0.8))
        self._groundplane_material = self.mjcf_model.asset.add(
            'material',
            name='groundplane',
            texrepeat=[2, 2],
            texuniform='true',
            reflectance=reflectance,
            texture=self._groundplane_texture)

        # Build groundplane.
        self._ground_geom = self._mjcf_root.worldbody.add(
            'geom',
            type='plane',
            name='groundplane',
            material=self._groundplane_material,
            size=list(size) + [groundplane_quad_size])

        # Add lighting
        self.mjcf_model.worldbody.add(
            'light',
            pos=(0, 0, 4),
            dir=(0, 0, -1),
            diffuse=(0.8, 0.8, 0.8),
            ambient=(0.2, 0.2, 0.2),
            specular=(.3, .3, .3),
            directional='true',
            castshadow='false')

        # Always initialize the free camera so that it points at the origin.
        self.mjcf_model.statistic.center = (3., 0., 0.)

    def attach_offset(self, entity, offset, attach_site=None):
        """Attaches another entity at a position offset from the attachment site.

        Args:
        entity: The `Entity` to attach.
        offset: A length 3 array-like object representing the XYZ offset.
        attach_site: (optional) The site to which to attach the entity's model.
            If not set, defaults to self.attachment_site.
        Returns:
        The frame of the attached model.
        """
        frame = self.attach(entity, attach_site=attach_site)
        frame.pos = offset
        return frame


class CustomArena(composer.Arena):
    """Suite-specific subclass of the custom Composer arena."""

    def _build(self, *args, **kwargs):
        """Initializes this arena.
        name: (optional) A string, the name of this arena. If `None`, use the
            model name defined in the MJCF file.
        """
        super()._build(*args, **kwargs)

    def attach_offset(self, entity, offset, attach_site=None):
        """Attaches another entity at a position offset from the attachment site.

        Args:
        entity: The `Entity` to attach.
        offset: A length 3 array-like object representing the XYZ offset.
        attach_site: (optional) The site to which to attach the entity's model.
            If not set, defaults to self.attachment_site.
        Returns:
        The frame of the attached model.
        """
        frame = self.attach(entity, attach_site=attach_site)
        frame.pos = offset
        return frame

    def add_entity(self, entity):
        if entity.is_free:
            return self.add_free_entity(entity)
        else:
            return self.attach(entity)
