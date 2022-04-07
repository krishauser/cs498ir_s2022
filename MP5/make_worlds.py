from klampt import WorldModel
from klampt.model.create import primitives

def make_table(world,width,depth,height,leg_width=0.05,color=(0.4,0.3,0.2)):
    table_width = width
    table_depth = depth
    table_height = height
    lw = leg_width
    table_surf = primitives.box(table_width,table_depth,lw,center=(0,0,table_height-lw*0.5),world=world,mass=float('inf'))
    table_legs = []
    table_legs.append(primitives.box(lw,lw,table_height-lw,center=(-table_width*0.5+lw,-table_depth*0.5+lw,(table_height-lw)*0.5),world=world,mass=float('inf')))
    table_legs.append(primitives.box(lw,lw,table_height-lw,center=(table_width*0.5-lw,-table_depth*0.5+lw,(table_height-lw)*0.5),world=world,mass=float('inf')))
    table_legs.append(primitives.box(lw,lw,table_height-lw,center=(-table_width*0.5+lw,table_depth*0.5-lw,(table_height-lw)*0.5),world=world,mass=float('inf')))
    table_legs.append(primitives.box(lw,lw,table_height-lw,center=(table_width*0.5-lw,table_depth*0.5-lw,(table_height-lw)*0.5),world=world,mass=float('inf')))
    table_surf.appearance().setColor(*color)
    table_surf.setName("table_top")
    for i,l in enumerate(table_legs):
        l.setName("table_leg_"+str(i+1))
        l.appearance().setColor(*color)
    return [table_surf] + table_legs

def make_rectangular_box(world,width,depth,height,thickness=0.02,bottom_height=0,color=(0.3,0.6,0.2)):
    bottom = primitives.box(width,depth,thickness,center=(0,0,thickness*0.5),world=world,mass=float('inf'))
    sides = []
    sides.append(primitives.box(width,thickness,height,center=(0,-depth*0.5+thickness*0.5,height*0.5),world=world,mass=float('inf')))
    sides.append(primitives.box(width,thickness,height,center=(0,depth*0.5-thickness*0.5,height*0.5),world=world,mass=float('inf')))
    sides.append(primitives.box(thickness,depth,height,center=(-width*0.5+thickness*0.5,0,height*0.5),world=world,mass=float('inf')))
    sides.append(primitives.box(thickness,depth,height,center=(width*0.5-thickness*0.5,0,height*0.5),world=world,mass=float('inf')))
    items = [bottom] + sides
    bottom.setName("box_bottom")
    for i,l in enumerate(sides):
        l.setName("box_side_"+str(i+1))
    for l in items:
        l.appearance().setColor(*color)
    return items

if __name__ == '__main__':
    world = WorldModel()
    world.readFile("../data/terrains/plane.env")
    make_table(world,0.4,0.6,0.5)
    world.saveFile("table.xml","table/")

    world = WorldModel()
    world.readFile("../data/terrains/plane.env")
    make_rectangular_box(world,0.6,0.4,0.1)
    world.saveFile("box.xml","box/")
