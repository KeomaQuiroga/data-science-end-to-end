# Fundamentos de Ciencia de Datos

Repositorio base del curso.

Aquí viven los laboratorios y los PSets, construidos alrededor de un solo proyecto de ciencia de datos que cada estudiante define y lleva de principio a fin durante las 16 semanas. Es la fuente de verdad: todo tu trabajo del semestre va aquí, no en otro lado.

## Antes de empezar

No clones ni hagas push a este repositorio directamente. Cada estudiante trabaja en su propio fork:

1. Arriba a la derecha, clic en **"Fork"** y créalo en tu propia cuenta de GitHub.
2. Todo tu trabajo del semestre (laboratorios, PSets) va en tu fork, no aquí.

## Sobre el ambiente de desarrollo

Las primeras dos semanas del curso son de formulación y diseño del proyecto: no requieren código ni un ambiente de desarrollo todavía. Desde la Semana 3, Sesión 1 ("Contenedores para pipelines de proyectos"), cada laboratorio produce código real: empaquetas el ambiente de tu proyecto con Docker (Semana 3, Sesión 1) y aprovisionas su almacenamiento con Terraform sobre Google Cloud (Semana 3, Sesión 2). Cada carpeta de `laboratorios/` que lo requiera trae sus propios requisitos e instrucciones de instalación en su `README.md`; no hay un ambiente único para todo el curso, porque cada estudiante empaqueta el ambiente de su propio proyecto, no uno compartido.

## Estructura del repositorio

```
laboratorios/           Laboratorio de cada clase, uno por carpeta
  semana-02-sesion-02/
  semana-03-sesion-01/
  semana-03-sesion-02/
  ...
psets/                  Tus entregas de PSet, una carpeta por PSet
  pset-01/
  ...
```

Cada carpeta de `laboratorios/` y `psets/` tiene su propio `README.md` con las instrucciones de esa entrega.

## Cómo actualizar tu fork cada semana

Cuando se agregue una semana nueva, sincroniza tu fork:

1. En la página de tu fork en GitHub: botón **"Sync fork" → "Update branch"**.
2. En tu copia local o Codespace, haz `git pull` para traer esos cambios.

Por terminal, en vez del botón:
```
git fetch upstream
git merge upstream/master
```
(la primera vez, agrega el remoto: `git remote add upstream https://github.com/ErickNaunay/fundamentos-ciencia-datos.git`)

## Cómo trabajar aquí

- Haz commit de tu avance seguido, con mensajes que digan qué hiciste, no `"cambios"` o `"fix"`.
- Cada entrega (laboratorio, PSet) vive en su propia carpeta con su propio `README.md`: revísalo antes de empezar.
