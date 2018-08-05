---
layout: archive
title: "Projects"
permalink: /projects/
author_profile: true
---

Here are some things I've been working one.

{% include base_path %}

{% for post in site.projects reversed %}
  {% include archive-single.html %}
{% endfor %}