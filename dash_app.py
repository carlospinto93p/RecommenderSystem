# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np

max_scores = 40
# Load data:
mangas = pd.read_csv('mangas_v2.csv')
scores = pd.read_csv('scores_v2.csv')
ratings = pd.merge(mangas, scores, on='manga_id')
ratings = ratings[['manga_id', 'user', 'score']]
corr_mangas = pd.read_csv('corr_pearson_mangas.csv', header=0, index_col=0)
pivot_users = ratings.pivot_table(index=['manga_id'], columns=['user'], values='score')
pivot_mangas = ratings.pivot_table(index=['user'], columns=['manga_id'], values='score')


# Functions:
def recomendations_by_item_similarity_for_new_user(new_user, filtered=False, subtype='specifity'):
    user_recomendation = pd.Series()
    for manga_index in range(len(new_user)):
        similar_mangas = corr_mangas[str(new_user.index[manga_index])].dropna()
        similar_mangas = similar_mangas.drop(new_user.index, errors='ignore')
        score = new_user.values[manga_index]
        similar_mangas = similar_mangas.map(lambda x: x * score if score > 5 else (x - 5) * score)
        user_recomendation = user_recomendation.append(similar_mangas)
    if subtype == 'popularity':
        user_recomendation = user_recomendation.groupby(user_recomendation.index).sum()
    elif subtype == 'specifity':
        user_recomendation = user_recomendation.groupby(user_recomendation.index).mean()
    user_recomendations = pd.DataFrame(user_recomendation)
    user_recomendations.columns = ['recomended_score']
    user_recomendations['manga_id'] = user_recomendations.index
    user_recomendations.index = np.arange(user_recomendations.shape[0])
    df_user_recomendation = pd.merge(user_recomendations, mangas, on=['manga_id'], how='inner')
    recomendations = df_user_recomendation.sort_values(by=['recomended_score', 'mean_score'], ascending=False)
    recomendations.index = np.arange(recomendations.shape[0])
    return recomendations


def recomendations_by_user_similarity_for_new_user(new_user, subtype='specifity'):
    similar_users_by_pearson = pivot_users.corrwith(other=new_user, method='pearson').dropna()
    similar_users_by_spearman = pivot_users.corrwith(other=new_user, method='spearman').dropna()
    # La similaridad entre usuarios según la ponderación entre los dos tipos de correlaciones.
    similar_users = 0.7 * similar_users_by_spearman + 0.3 * similar_users_by_pearson
    # Tabla de usuarios similares:
    df_similar_users = pd.DataFrame(similar_users)
    df_similar_users.columns = ['Similarity']
    # Me quedo solo con los usuarios con puntuación de similaridad positiva.
    df_similar_users = df_similar_users[df_similar_users['Similarity'] > 0]
    # Empiezo a construir las puntuaciones de las recomendaciones:
    recomendations_by_similar_users = pd.Series()
    for i in range(df_similar_users.shape[0]):
        current_user = df_similar_users.index[i]
        user_similarity = df_similar_users.values[i][0]
        # Hago una serie para tratar los datos de cada usuario similar:
        series_current_user = pd.Series(ratings[ratings['user'] == current_user].score)
        series_current_user.index = ratings[ratings['user'] == current_user].manga_id
        # Elimino de esta serie los productos que ya ha consumido el objetivo:
        series_current_user = series_current_user.drop(new_user.index, errors='ignore')
        # Multiplico las puntuaciones por la similaridad del usuario. Notemos que penalizo las
        # notas suspensas (aunque no con mucha fuerza)
        series_current_user = series_current_user.map(
            lambda x: x * user_similarity if x > 5 else (x - 5) * user_similarity)
        recomendations_by_similar_users = recomendations_by_similar_users.append(series_current_user)
    # Función de agregación: por popularidad o por especifidad:
    if subtype == 'popularity':
        recomendations_by_similar_users = recomendations_by_similar_users.groupby(
            recomendations_by_similar_users.index).sum()
    elif subtype == 'specifity':
        recomendations_by_similar_users = recomendations_by_similar_users.groupby(
            recomendations_by_similar_users.index).mean()
    # Construyo el dataframe para la presentación de resultados:
    recomendations = pd.DataFrame(recomendations_by_similar_users)
    recomendations.columns = ['recomended_score']
    recomendations['manga_id'] = recomendations.index
    recomendations.index = np.arange(recomendations.shape[0])
    recomendations = pd.merge(recomendations, mangas, on=['manga_id'], how='inner')
    # Recomendaciones:
    recomendations = recomendations.sort_values(by=['recomended_score', 'mean_score'], ascending=False)
    recomendations.index = np.arange(recomendations.shape[0])
    return recomendations


def recomendations_by_item_similarity_for_given_user(user_id, filtered=False, subtype='specifity'):
    user = pivot_mangas.iloc[user_id].dropna()
    return recomendations_by_item_similarity_for_new_user(user, filtered=filtered, subtype=subtype)


def recomendations_by_user_similarity_for_given_user(user_id, subtype='specifity'):
    user = pivot_mangas.iloc[user_id].dropna()
    return recomendations_by_user_similarity_for_new_user(user, subtype=subtype)


def get_recomendations(user, main_type='user_similarity', subtype='specifity', reduced_dtb=False):
    if main_type == 'user_similarity':
        return recomendations_by_user_similarity_for_new_user(user, subtype=subtype)
    elif main_type == 'item_similarity':
        return recomendations_by_item_similarity_for_new_user(user, filtered=reduced_dtb, subtype=subtype)
    else:
        raise Exception("You must select between 'user_similarity' or 'item_similarity'.")


def validate_user(user):
    flag = True
    for manga_name in user.keys():
        if manga_name not in mangas['manga_name'].values:
            print(f'The manga {manga_name} is not on the manga database.')
            flag = False
    if flag:
        print('Data introduced is ok.')
    return flag


def prepare_new_user(user):
    manga_dict = {}
    for manga_name in user.keys():
        manga_id = mangas[mangas['manga_name'] == manga_name].manga_id.iloc[0]
        manga_dict[manga_id] = user[manga_name]
    new_user = pd.Series(manga_dict)
    return new_user


# ----

colors = {
    'background': '#111111',
    'placeholder': '#777777',
    'text': '#DDDDDD'
}

# Model:

df = pd.read_csv('reviews_scores.csv')
recomendations = pd.DataFrame()


def generate_table(dataframe, max_rows=15):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +
        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))],
        style={'margin': '2em auto', 'float': 'none'})


def generate_interactive_table(dataframe, init_col, max_rows=15):
    return [
        html.Div(children=[
            html.Div('Ordenar por:', style={'width': '25%', 'float': 'left', 'marginBottom': '13px',
                                            'textAlign': 'right', 'paddingRight': '20px'}),
            html.Div(
                dcc.Dropdown(
                    id='select_column',
                    options=[{'label': col, 'value': col} for col in dataframe.columns],
                    value=init_col,
                    clearable=False,
                    multi=True,
                    style={'color': colors['placeholder']}
                ),
                style={'width': '70%', 'float': 'right'}),
            dcc.Checklist(
                id='order_type',
                options=[{'label': 'Ascending', 'value': 1}],
                values=[], style={'float': 'right'}
            ),
            html.Br(style={'clear': 'both'}),
        ], style={'width': '50%', 'margin': 'auto'}),
        html.Div(id='df_rec', children=generate_table(dataframe, max_rows=max_rows),
                 style={'marginTop': '-20`px'})
    ]


dropdown_div_style = {'float': 'left', 'marginLeft': '40px', 'width': '30%', 'display': 'none'}
slider_div_style = {'float': 'left', 'marginLeft': '20px', 'width': '12%', 'display': 'none', 'color': colors['text']}
row_manga_score_style = {'margin': 'auto', 'width': '75%', 'display': 'none'}
remove_manga_style = {'color': colors['text'], 'marginLeft': '50px', 'display': 'none', 'float': 'left'}


def generate_score_field(id1):
    field = (
        html.Div([
            dcc.Dropdown(
                options=[{'label': manga_name, 'value': manga_name} for manga_name in
                         mangas.manga_name.values],
                value='.', style={'color': colors['placeholder']}, id='dr' + str(id1)
            )
        ], style=dropdown_div_style, id='d' + str(id1)),
        html.Div([
            dcc.Slider(
                min=1,
                max=10,
                marks={i: str(i) for i in range(1, 11)},
                value=5, id='sl' + str(id1)
            )], style=slider_div_style, id='s' + str(id1))
    )
    return field


def generate_div_init():
    return html.Div(style={'marginBottom': '30px'})


def generate_div_end():
    return html.Br(style={'clear': 'both'})


def generate_row_scores(id1, complete=False):
    score_field_1 = generate_score_field(2*id1)
    elems = [generate_div_init(),
             score_field_1[0],
             score_field_1[1],
             generate_div_end()]
    if complete:
        score_field_2 = generate_score_field(2*id1+1)
        elems = [generate_div_init(),
                 score_field_1[0],
                 score_field_1[1],
                 score_field_2[0],
                 score_field_2[1],
                 generate_div_end()]
    return html.Div(elems, style=row_manga_score_style, id='r'+str(id1))


def generate_html_scores(n_scores):
    if n_scores == 1:
        return [generate_row_scores(1, False)]
    if n_scores == 2:
        return [generate_row_scores(1, True)]
    return [generate_row_scores(i, True) for i in range(n_scores//2-1)] + \
           [generate_row_scores(n_scores//2-1, False) if (n_scores % 2) == 1 else generate_row_scores(n_scores//2-1,
                                                                                                      True)]


def generate_scores_output(max_scores_):
    outs = []
    for i in range(max_scores_):
        outs.append(Output('d' + str(i), 'style'))
        outs.append(Output('s' + str(i), 'style'))
    for i in range(max_scores_//2):
        outs.append(Output('r' + str(i), 'style'))
    return outs


def generate_recs_states(max_scores_):
    states = []
    # Append dropdowns (manga_names)
    for i in range(max_scores_):
        states.append(State('dr' + str(i), 'value'))
    # Append sliders (manga_scores)
    for i in range(max_scores_):
        states.append(State('sl' + str(i), 'value'))
    return states


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# children is by default the first argument and it is common to be omitted.
app.layout = \
    html.Div(style={'backgroundColor': colors['background'], 'align': 'center', 'color': colors['text'],
                    'margin': '-8px', 'minHeight': '100vh'},
             children=[
                 html.H1(
                     children='Sistema de recomendación de mangas',
                     style={
                         'paddingTop': '2rem',
                         'textAlign': 'center'
                     }
                 ),

                 html.Div(children=(html.P('Introduce una lista de mangas y puntuaciones asignadas, y los parámetros'
                                    + ' para elegir el modelo de recomendación.'),
                                    html.P('Advertencia: generar recomendaciones por similaridad entre perfiles de' +
                                    ' usuario puede tardar varios minutos.')
                                    ), style={'textAlign': 'center', 'marginBottom': '30px'}),

                 html.Div([
                     html.Button('Add manga score', id='add_manga', style={'color': colors['text'],
                                 'marginLeft': '30px', 'float': 'left'}),
                     html.Button('Remove manga score', id='remove_manga', style={'color': colors['text'],
                                 'marginLeft': '50px', 'display': 'none', 'float': 'left'}),
                     html.Br(style={'clear': 'both'})
                 ], style={'width': '40%', 'margin': 'auto'}),

                 html.Div(generate_html_scores(max_scores), id='scores_div'),
                 # -----
                 html.Div([
                    html.Div([
                        html.H5('Recomendar en base a:'),
                        dcc.RadioItems(
                            id='recomendation_params_1',
                            options=[
                                {'label': 'Similaridad entre perfiles de usuario', 'value': 'user_similarity'},
                                {'label': 'Similaridad entre mangas', 'value': 'item_similarity'}
                            ],
                            value='user_similarity',
                            style={'color': colors['text']}
                        ),
                    ], style={'float': 'left', 'marginLeft': '60px', 'width': '30%'}),

                    html.Div([
                        html.H5('Subtipo de recomendación:'),
                        dcc.RadioItems(
                            id='recomendation_params_2',
                            options=[
                                {'label': 'Popularidad', 'value': 'popularity'},
                                {'label': 'Especifidad', 'value': 'specifity'}
                            ],
                            value='popularity',
                            style={'color': colors['text']}
                        ),
                    ], style={'float': 'left', 'marginLeft': '20px', 'width': '30%'}),

                    html.Div([
                        html.Button('Generar recomendaciones', id='submit_button',
                                    style={'color': colors['text']}),
                    ], style={'float': 'left', 'marginLeft': '20px', 'marginTop': '60px', 'width': '20%'})

                 ], style={'margin': 'auto', 'width': '80%'}, id='params'),

                 html.Br(style={'clear': 'both'}),

                 html.Div([], style={'marginBottom': '20px'}),

                 html.Div([

                    html.H3('Resultados:', style={'marginBottom': '20px', 'textAlign': 'center'}),

                    dcc.Loading(id='loading', children=[
                        html.Div(id='df_rec'),
                        dcc.Dropdown(
                            id='select_column',
                            options=[{'label': col, 'value': col} for col in mangas.columns],
                            value='n',
                            style={'width': '50%', 'color': colors['text']}
                        ), dcc.Checklist(
                            id='order_type',
                            options=[
                                {'label': 'Ascending', 'value': 1},
                            ],
                            values=[], style={'float': 'right'})
                    ], type='default')

                 ], style={'margin': 'auto', 'width': '80%'}, id='result_view'),

                 html.Div('Carlos Pinto Pérez',
                          style={'paddingTop': '60px', 'width': '95%', 'backgroundColor': colors['background'],
                                 'color': colors['text'], 'textAlign': 'right', 'paddingBottom':'20px'},
                          id='bottom_page')
             ])


@app.callback(
    [Output('remove_manga', 'style')] + generate_scores_output(max_scores),
    [Input('add_manga', 'n_clicks'), Input('remove_manga', 'n_clicks')])
def generate_scores(n_add, n_rem):
    basic_list = [remove_manga_style]
    for i in range(max_scores):
        basic_list.append(dropdown_div_style.copy())
        basic_list.append(slider_div_style.copy())
    for i in range(max_scores // 2):
        basic_list.append(row_manga_score_style.copy())
    use_list = basic_list.copy()
    if n_add is None:
        n_scores = 1
    elif n_rem is None:
        n_scores = n_add+1
    else:
        n_scores = n_add-n_rem+1
    if n_scores > max_scores:
        n_scores = max_scores
    # Remove manga button style (if appears or not)
    if n_scores > 1:
        use_list[0]['display'] = 'block'
    else:
        use_list[0]['display'] = 'none'
    # Show or not the required fields to introduce manga scores:
    for i in range(n_scores):
        use_list[2*i+1]['display'] = 'block'
        use_list[2*i+2]['display'] = 'block'
    for i in range((n_scores-1)//2 + 1):
        use_list[2*max_scores+1 + i]['display'] = 'block'
    return use_list


@app.callback(
    Output('loading', 'children'),
    [Input('submit_button', 'n_clicks')],
    generate_recs_states(max_scores) + [State('recomendation_params_1', 'value'),
                                        State('recomendation_params_2', 'value')])
def generate_recs(click, *args):
    """Need to put load time to this."""
    if click is None:
        return generate_interactive_table(mangas, 'n')
    print('Started...')
    keys = [m_name for m_name in args[:max_scores]]
    values = [m_val for m_val in args[max_scores:]]
    # Validate keys:
    user_k = {key: values[i] for i, key in enumerate(keys) if key in mangas.manga_name.values}
    print(user_k)
    user = prepare_new_user(user_k)
    print(user)
    # Params 1: ('user_similarity', 'item_similarity')
    param_1 = args[-2]
    # Params 2: ('popularity', 'specifity')
    param_2 = args[-1]
    print(param_1)
    print(param_2)
    global recomendations
    recomendations = get_recomendations(user, main_type=param_1, subtype=param_2)
    print('Ended.')
    return generate_interactive_table(recomendations, 'recomended_score')


@app.callback(
    Output('df_rec', 'children'),
    [Input('select_column', 'value'), Input('order_type', 'values')],
    [State('submit_button', 'n_clicks')])
def update_df(column_chosen, order_type_val, df_mangas):
    if len(order_type_val) == 0:
        order_type_val = False
    else:
        order_type_val = True
    if isinstance(column_chosen, str):
        reps = 1
    else:
        reps = len(column_chosen)
    if df_mangas is None:
        if column_chosen == 'n' or len(column_chosen) == 0:
            return generate_table(mangas)
        else:
            return generate_table(mangas.sort_values(column_chosen, ascending=[order_type_val]*reps))
    else:
        if column_chosen == 'n' or len(column_chosen) == 0:
            return generate_table(recomendations)
        else:
            return generate_table(recomendations.sort_values(column_chosen, ascending=[order_type_val]*reps))


if __name__ == '__main__':
    app.run_server(debug=False)
