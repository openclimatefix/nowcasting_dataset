# get satelite image

import staticmaps

osm = staticmaps.tile_provider_OSM

context = staticmaps.Context()
context.set_tile_provider(osm)

frankfurt = staticmaps.create_latlng(51.75, -1.25)
newyork = staticmaps.create_latlng(52.63, 1.29)

colour = staticmaps.color.Color()

# context.add_object(staticmaps.Line([frankfurt, newyork], color=staticmaps.BLUE, width=4))
context.add_object(staticmaps.Marker(frankfurt, color=staticmaps.GREEN, size=12))
context.add_object(staticmaps.Marker(newyork, color=staticmaps.RED, size=12))

print(context.determine_center_zoom(1000,1000))
print(context._tile_provider.tile_size())


image = context.render_pillow(1000, 1000)
image.save("frankfurt_newyork.pillow.png")