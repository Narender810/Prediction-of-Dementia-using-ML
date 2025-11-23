def get_footer():
    footer_style = """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f0f0f0;
        text-align: center;
        padding: 10px 0;
        margin: 0; 
        border-top: 1px solid #ccc; 
    }
    .footer a {
        color: blue;
        text-decoration: none;
    }
    </style>
    """
    
    footer_content = "<div class='footer'><p>© 2024 team. All rights reserved. | Icons made by <a href='#'>team</a>></p></div>"

    return footer_style + footer_content
