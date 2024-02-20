import tkinter as tk
import tkinter.simpledialog
import numpy as np

class Draw_Graph:
    def __init__(self, master, directed=False, valued=False):
        self.master = master
        self.vertices = []
        self.edges = []
        self.directed = directed
        self.valued=valued
        #self.oriented = oriented  # Set the oriented attribute

        self.canvas = tk.Canvas(self.master, width=600, height=600, bg='white')
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.add_vertex)
        self.canvas.bind('<Button-3>', self.add_edge)

        self.matrix_label = tk.Label(self.master, text='')

        self.reset = tk.Button(self.master, text="reset", command=self.reset_fun)
        self.reset.pack()

        self.add_vertex_btn = tk.Button(self.master, text="ajouter les sommets", command=self.bind_vetrex )
        self.add_vertex_btn.pack()

        
        self.show_matrix_button = tk.Button(self.master, text='enregistrer dans matrice.txt', command=self.show_matrix)
        self.show_matrix_button.pack()


    def bind_vetrex(self):
        self.canvas.bind('<Button-1>', self.add_vertex)
        self.canvas.bind('<Button-3>', self.add_edge)
    
    def reset_fun(self):
        self.canvas.delete("all")
        self.canvas.bind('<Button-1>', self.add_vertex)
        self.canvas.bind('<Button-3>', self.add_edge)
        self.vertices = []
        self.edges = []
        

    def add_vertex(self, event):
        x, y = event.x, event.y
        vertex = self.canvas.create_oval(x-10, y-10, x+10, y+10, fill='white')
        vertex_label = self.canvas.create_text(x, y, text=str(len(self.vertices)))
        self.vertices.append((x, y, vertex, vertex_label))


    def add_edge(self, event):
        x, y = event.x, event.y
        vertex = self.get_vertex_at_position(x, y)
        if vertex:
            if self.valued:
                self.canvas.bind('<Button-1>', self.add_edge_value)
                self.temp_edge = (x, y, vertex)
            if not self.valued:
                self.canvas.bind('<Button-1>', self.add_edge_value)
                self.temp_edge = (x, y, vertex)
        

    def add_edge_value(self, event):
        x, y = event.x, event.y
        vertex = self.get_vertex_at_position(x, y)
        if vertex:
            if not self.directed and self.valued:
                # If the graph is non-oriented, unbind the right mouse click event
                self.canvas.unbind('<Button-1>')
                value = self.get_edge_value()
                edge = self.canvas.create_line(self.temp_edge[0], self.temp_edge[1], x, y)
                edge_label = tk.Label(self.master, text=str(value))
                self.canvas.create_window((self.temp_edge[0] + x) / 2 + 10, (self.temp_edge[1] + y) / 2, window=edge_label)
                self.edges.append((self.temp_edge[2], vertex, value, edge, edge_label))
                self.temp_edge = None
            if self.directed and self.valued:
                # If the graph is non-oriented, unbind the right mouse click event
                self.canvas.unbind('<Button-1>')
                value = self.get_edge_value()
                edge = self.canvas.create_line(self.temp_edge[0], self.temp_edge[1], x, y,arrow=tk.LAST)
                edge_label = tk.Label(self.master, text=str(value))
                self.canvas.create_window((self.temp_edge[0] + x) / 2 + 10, (self.temp_edge[1] + y) / 2, window=edge_label)
                self.edges.append((self.temp_edge[2], vertex, value, edge, edge_label))
                self.temp_edge = None
            if self.directed and not self.valued:
                # If the graph is non-oriented, unbind the right mouse click event
                self.canvas.unbind('<Button-1>')
                #value = self.get_edge_value()
                edge = self.canvas.create_line(self.temp_edge[0], self.temp_edge[1], x, y,arrow=tk.LAST)
                #edge_label = tk.Label(self.master, text=str(value))
                #self.canvas.create_window((self.temp_edge[0] + x) / 2 + 10, (self.temp_edge[1] + y) / 2, window=edge_label)
                self.edges.append((self.temp_edge[2], vertex, 1, edge, None))
                self.temp_edge = None
            if not self.directed and not self.valued:
                # If the graph is non-oriented, unbind the right mouse click event
                self.canvas.unbind('<Button-1>')
                #value = self.get_edge_value()
                edge = self.canvas.create_line(self.temp_edge[0], self.temp_edge[1], x, y)
                #edge_label = tk.Label(self.master, text=str(value))
                #self.canvas.create_window((self.temp_edge[0] + x) / 2 + 10, (self.temp_edge[1] + y) / 2, window=edge_label)
                self.edges.append((self.temp_edge[2], vertex, 1, edge, None))
                self.temp_edge = None




    def get_edge_value(self):
        value = tkinter.simpledialog.askinteger("ajouter la valeur d'arc", 'enter une valeur:')
        return value


    def get_vertex_at_position(self, x, y):
        for vertex in self.vertices:
            if (x-10) <= vertex[0] <= (x+10) and (y-10) <= vertex[1] <= (y+10):
                return vertex[2]
        return None

    def show_matrix(self):
        #print(np.array(self.vertices),"\n",np.array(self.edges))
        num_vertices = len(self.vertices)
        if self.valued:
            matrix = [[float("inf") for _ in range(num_vertices)] for _ in range(num_vertices)]
        else:
            matrix = [[0 for _ in range(num_vertices)] for _ in range(num_vertices)]
        if not self.directed:
            for i in range(len(matrix)):
                matrix[i][i]=0
        for i in range(num_vertices):
            for j in range(num_vertices):
                for edge in self.edges:
                    if self.directed:
                        if (self.vertices[i][2], self.vertices[j][2]) == edge[:2]:
                            matrix[i][j] = edge[2]
                        """elif (self.vertices[j][2], self.vertices[i][2]) == edge[:2]:
                            matrix[i][j] = edge[2]"""
                    else:
                        if (self.vertices[i][2], self.vertices[j][2]) == edge[:2]:
                            matrix[i][j] = edge[2]
                        elif (self.vertices[j][2], self.vertices[i][2]) == edge[:2]:
                            matrix[i][j] = edge[2]
        print(np.array(matrix))
        with open("matrice.txt","w") as f:
            for ln in matrix:
                for cl,j in zip(ln,range(len(matrix))):
                    f.write(str(cl))
                    if j!=len(matrix)-1:
                        f.write(",")
                f.write("\n")
        return matrix
        """matrix_str = '\n'.join([' '.join([str(cell) for cell in row]) for row in matrix])
        self.matrix_label['text'] = matrix_str"""
"""
root = tk.Tk()
graph = Draw_Graph(root)
root.mainloop()"""

