import streamlit as st

def show_login_page():
    html_code = """
        <style>
            /* Hide Streamlit elements */
            header, footer, #MainMenu, [data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
            .block-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }
            .stApp {
                background: linear-gradient(135deg, #0b1120 0%, #0f172a 50%, #020617 100%) !important;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            
            /* Full screen container */
            .login-wrapper {
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 1rem;
            }

            /* Glass Login Card */
            .login-card {
                background: rgba(17, 24, 39, 0.6);
                backdrop-filter: blur(16px);
                -webkit-backdrop-filter: blur(16px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 20px;
                padding: 2.5rem;
                width: 100%;
                max-width: 420px;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.05) inset;
                color: #f8fafc;
                text-align: center;
                animation: fadeInCard 0.6s ease-out forwards;
                opacity: 0;
                transform: translateY(20px);
            }

            @keyframes fadeInCard {
                to { opacity: 1; transform: translateY(0); }
            }

            /* Branding */
            .brand-icon {
                font-size: 2.5rem;
                margin-bottom: 1rem;
                display: inline-block;
                background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: pulseIcon 2.5s infinite ease-in-out alternate;
            }
            
            @keyframes pulseIcon {
                from { transform: scale(1); filter: drop-shadow(0 0 5px rgba(59, 130, 246, 0.3)); }
                to { transform: scale(1.05); filter: drop-shadow(0 0 15px rgba(139, 92, 246, 0.6)); }
            }

            .brand-title {
                font-size: 1.5rem;
                font-weight: 700;
                letter-spacing: -0.025em;
                margin-bottom: 0.25rem;
            }
            .brand-subtitle {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: #94a3b8;
                margin-bottom: 2rem;
                font-weight: 600;
            }

            .welcome-text {
                font-size: 1.125rem;
                font-weight: 600;
                margin-bottom: 0.25rem;
            }
            .welcome-sub {
                font-size: 0.875rem;
                color: #94a3b8;
                margin-bottom: 1.5rem;
            }

            /* Form Elements */
            .input-group {
                text-align: left;
                margin-bottom: 1.25rem;
                position: relative;
            }
            .input-group label {
                display: block;
                font-size: 0.875rem;
                font-weight: 500;
                color: #cbd5e1;
                margin-bottom: 0.5rem;
            }
            .input-field {
                width: 100%;
                background: rgba(15, 23, 42, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 0.75rem 1rem;
                color: white;
                font-size: 0.95rem;
                transition: all 0.2s ease;
                outline: none;
                box-sizing: border-box;
            }
            .input-field:focus {
                border-color: #3b82f6;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
                background: rgba(15, 23, 42, 1);
            }
            
            /* Password Toggle Eye */
            .eye-toggle {
                position: absolute;
                right: 1rem;
                top: 2.25rem;
                color: #94a3b8;
                cursor: pointer;
                background: none;
                border: none;
                padding: 0;
            }
            .eye-toggle:hover { color: #f8fafc; }

            /* Options row */
            .options-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.875rem;
                margin-bottom: 1.5rem;
            }
            .remember-me {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                color: #cbd5e1;
                cursor: pointer;
            }
            .remember-me input {
                accent-color: #3b82f6;
                width: 1rem;
                height: 1rem;
                cursor: pointer;
            }
            .forgot-link {
                color: #3b82f6;
                text-decoration: none;
                font-weight: 500;
                transition: color 0.2s;
            }
            .forgot-link:hover { color: #60a5fa; text-decoration: underline; }

            /* Submit Button */
            .submit-btn {
                width: 100%;
                background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
                color: white;
                font-weight: 600;
                font-size: 1rem;
                padding: 0.875rem;
                border-radius: 12px;
                border: none;
                cursor: pointer;
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 0.5rem;
                box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39);
                transition: all 0.2s ease;
                margin-bottom: 1.5rem;
            }
            .submit-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5);
                background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
            }
            .submit-btn:active {
                transform: translateY(0);
            }

            /* Divider */
            .divider {
                display: flex;
                align-items: center;
                text-align: center;
                color: #64748b;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                margin-bottom: 1.5rem;
            }
            .divider::before, .divider::after {
                content: '';
                flex: 1;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            .divider::before { margin-right: 0.75rem; }
            .divider::after { margin-left: 0.75rem; }

            /* Create Account */
            .create-account {
                font-size: 0.875rem;
                color: #cbd5e1;
            }
            .create-account a {
                color: #3b82f6;
                text-decoration: none;
                font-weight: 600;
                margin-left: 0.25rem;
                transition: color 0.2s;
            }
            .create-account a:hover { color: #60a5fa; text-decoration: underline; }

            /* Error message container */
            .error-message {
                display: none;
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: #fca5a5;
                padding: 0.75rem;
                border-radius: 10px;
                font-size: 0.875rem;
                margin-bottom: 1.5rem;
                text-align: left;
                animation: fadeInCard 0.3s ease;
            }
        </style>

        <div class="login-wrapper">
            <div class="login-card">
                <div class="brand-icon">🧠</div>
                <div class="brand-title">Sentiment Analysis</div>
                <div class="brand-subtitle">Enterprise Edition</div>

                <div class="welcome-text">Welcome back</div>
                <div class="welcome-sub">Sign in to your admin account</div>

                <div id="errorBox" class="error-message">
                    ⚠️ Invalid username or password.
                </div>

                <form action="" method="GET" target="_parent">
                    <!-- Critical for Streamlit navigation to recognize it as a login attempt -->
                    <input type="hidden" name="page" value="login_attempt">
                    
                    <div class="input-group">
                        <label for="username">Username</label>
                        <input type="text" id="username" name="username" class="input-field" placeholder="Enter your username" required autocomplete="username">
                    </div>
                    
                    <div class="input-group">
                        <label for="password">Password</label>
                        <input type="password" id="password" name="password" class="input-field" placeholder="••••••••" required autocomplete="current-password">
                        <button type="button" class="eye-toggle" onclick="const p=document.getElementById('password'); p.type=p.type==='password'?'text':'password'; this.innerHTML=p.type==='password'?'👁️':'🔒';">
                            👁️
                        </button>
                    </div>

                    <div class="options-row">
                        <label class="remember-me">
                            <input type="checkbox"> Remember me
                        </label>
                        <a href="#" class="forgot-link">Forgot password?</a>
                    </div>

                    <button type="submit" class="submit-btn" onclick="this.innerHTML='🚀 Signing In...'; this.style.opacity='0.8';">
                        🚀 Sign In
                    </button>
                </form>

                <div class="divider">OR</div>

                <div class="create-account">
                    New admin?<a href="?page=create_admin" target="_parent">Create an account</a>
                </div>
            </div>
        </div>

        <img src="x" onerror="
            if (window.parent.location.search.includes('error=1')) {
                var errBox = document.getElementById('errorBox');
                if (errBox) errBox.style.display = 'block';
                // Remove error from URL without refreshing, so it doesn't stay if the user refreshes
                var cleanUrl = window.parent.location.href.split('&error=1')[0].split('?error=1')[0];
                if (!cleanUrl.includes('page=login')) {
                    cleanUrl = '?page=login';
                }
                window.parent.history.replaceState({}, document.title, cleanUrl);
            }
        " style="display:none;">
    """
    
    import re
    html_clean = re.sub(r'^[ \t]+', '', html_code, flags=re.MULTILINE)
    st.markdown(html_clean, unsafe_allow_html=True)

def show_create_admin_page():
    # ── Streamlit layout overrides ───────────────────────────────────────────
    import re
    html_raw = """
        <style>
            header, footer, #MainMenu,
            [data-testid="stHeader"],
            [data-testid="stToolbar"]    { display: none !important; }
            .block-container             { padding: 2rem !important; max-width: 600px !important; margin: 0 auto; }
            .stApp                       { background: #070a12 !important; color: #f1f5f9; font-family: 'Inter', sans-serif; }
            /* Form styling overrides */
            div[data-testid="stForm"] {
                background: rgba(17, 24, 39, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 22px;
                padding: 2.5rem;
                box-shadow: 0 24px 64px -16px rgba(0,0,0,0.8), 0 0 80px -20px rgba(59,130,246,0.35);
            }
            .stTextInput>div>div>input {
                background: #0a0f1c; color: #f1f5f9; border-radius: 12px; border: 1.5px solid #1e293b; padding: 0.75rem;
            }
            .stButton>button {
                width: 100%; background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
                color: #fff; border: none; border-radius: 12px; padding: 0.75rem; font-weight: 600;
            }
            .stButton>button:hover {
                transform: translateY(-1px); border: none; color: #fff;
            }
        </style>
    """
    html_clean = re.sub(r'^[ \t]+', '', html_raw, flags=re.MULTILINE)
    st.markdown(html_clean, unsafe_allow_html=True)

    st.markdown("<div style='text-align: center; margin-bottom: 2rem;'><span style='font-size: 3rem;'>➕</span><h2 style='margin:0'>Add Administrator</h2><p style='color:#94a3b8; font-size: 0.9rem;'>Register a new administrator to manage platform analytics</p></div>", unsafe_allow_html=True)

    with st.form("create_admin_form"):
        username = st.text_input("New Username", placeholder="e.g. jsmith")
        password = st.text_input("New Password", type="password", placeholder="Minimum 6 characters")
        
        submitted = st.form_submit_button("Create Admin")
        
        if submitted:
            if not username or not password:
                st.error("❌ Please fill in both fields.")
            elif len(password) < 6:
                st.error("❌ Password must be at least 6 characters.")
            else:
                import bcrypt
                from backend.database import create_admin
                
                hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                success = create_admin(username, f"{username}@dashboard.ai", hashed)
                if success:
                    st.success(f"✅ Admin '{username}' created successfully!")
                else:
                    st.error(f"❌ Username '{username}' already exists.")

    st.markdown("<div style='text-align: center; margin-top: 1.5rem;'><a href='?page=login' style='color:#3b82f6; text-decoration:none; font-weight:600;'>Back to Login</a></div>", unsafe_allow_html=True)
